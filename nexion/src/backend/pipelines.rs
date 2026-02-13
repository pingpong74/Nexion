use ash::vk;
use std::{cell::UnsafeCell, collections::HashMap, path::PathBuf};

use crate::{
    Pipeline, PushConstantsDescription,
    backend::{device::InnerDevice, gpu_resources::ResourcePool},
};

use serde::{Deserialize, Serialize};

use crate::{ComputePipelineDescription, GeometryStage, RasterizationPipelineDescription};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    process::Command,
    sync::Arc,
    time::UNIX_EPOCH,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct ShaderCacheEntry {
    slang: String,
    spv: String,
    timestamp: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct PipelineSlot {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) bind_point: vk::PipelineBindPoint,
    pub(crate) push_constants_info: PushConstantsDescription,
}

pub(crate) struct InnerPipelineManager {
    pub(crate) desc_layout: vk::DescriptorSetLayout,
    pub(crate) pipelines: UnsafeCell<ResourcePool<PipelineSlot>>,
    pub(crate) device: Arc<InnerDevice>,
}

impl InnerPipelineManager {
    pub(crate) fn new(device: Arc<InnerDevice>) -> InnerPipelineManager {
        let cache_dir = Path::new(".cache");

        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir).expect("Failed to create cache directory");
            println!(".cache directory created");
        }

        return InnerPipelineManager {
            desc_layout: device.bindless_descriptors.layout,
            pipelines: UnsafeCell::new(ResourcePool::new()),
            device: device,
        };
    }

    fn compile_shader(path: &Path) -> PathBuf {
        let dst_path = Path::new(".cache").join(path.file_name().unwrap()).with_extension("spv");

        let output = Command::new("slangc")
            .arg(path)
            .arg("-o")
            .arg(&dst_path) // replaces .slang with .spv and also places the compiled shaders inside the .cache directory
            .output()
            .unwrap();

        if !output.status.success() {
            eprintln!("Failed to compile shader {:?}: {}", path, String::from_utf8_lossy(&output.stderr));
        } else {
            println!("Compiled shader {:?}", path);
        }

        return dst_path;
    }

    fn get_spv_code(path: &str) -> Vec<u32> {
        let dst_path = Self::compile_shader(Path::new(path));
        let bytes = fs::read(dst_path).unwrap();
        return bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    }

    fn create_shader_module(&self, path: &str) -> vk::ShaderModule {
        let shader = Self::get_spv_code(path);

        let module_create_info = vk::ShaderModuleCreateInfo::default().code(shader.as_slice());

        return unsafe { self.device.handle.create_shader_module(&module_create_info, None).expect("Failed to crate shader module") };
    }
}

//// Pipeline creation ////
impl InnerPipelineManager {
    pub(crate) fn create_raster_pipeline_data(&self, desc: &RasterizationPipelineDescription) -> Pipeline {
        let entry = std::ffi::CString::new("main").unwrap();

        let layouts = [self.desc_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .offset(desc.push_constants.offset)
            .size(desc.push_constants.size)
            .stage_flags(desc.push_constants.stage_flags.to_vk())];

        let layout_info = if desc.push_constants.size > 0 {
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push_ranges)
        } else {
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts)
        };

        let pipeline_layout = unsafe { self.device.handle.create_pipeline_layout(&layout_info, None).unwrap() };

        let mut shader_modules = Vec::new();
        let mut stages = Vec::new();

        let mut load_stage = |path: &str, stage: vk::ShaderStageFlags| {
            let code = Self::get_spv_code(path);
            let module = unsafe { self.device.handle.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).expect("Failed to create shader module") };
            shader_modules.push(module);
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(stage).module(module).name(&entry));
        };

        match &desc.geometry {
            GeometryStage::Classic { vertex_shader, .. } => {
                load_stage(vertex_shader, vk::ShaderStageFlags::VERTEX);
            }
            GeometryStage::Mesh { task_shader, mesh_shader } => {
                if let Some(task) = task_shader {
                    load_stage(task, vk::ShaderStageFlags::TASK_EXT);
                }
                load_stage(mesh_shader, vk::ShaderStageFlags::MESH_EXT);
            }
        }

        load_stage(desc.fragment_shader_path, vk::ShaderStageFlags::FRAGMENT);

        // ---------------- Fixed Function ----------------

        let vertex_bindings;
        let vertex_attributes;

        let (vertex_input, input_assembly) = match &desc.geometry {
            GeometryStage::Classic { vertex_input, topology, .. } => {
                (vertex_bindings, vertex_attributes) = vertex_input.to_vk();
                (
                    vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&vertex_bindings).vertex_attribute_descriptions(&vertex_attributes),
                    vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology.to_vk()),
                )
            }
            GeometryStage::Mesh { .. } => (vk::PipelineVertexInputStateCreateInfo::default(), vk::PipelineInputAssemblyStateCreateInfo::default()),
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(desc.polygon_mode.to_vk_flag())
            .cull_mode(desc.cull_mode.to_vk_flag())
            .front_face(desc.front_face.to_vk_flag())
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1).sample_shading_enable(false);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_stencil.depth_test_enable)
            .depth_write_enable(desc.depth_stencil.depth_write_enable)
            .depth_compare_op(desc.depth_stencil.depth_compare_op.to_vk())
            .depth_bounds_test_enable(false)
            .stencil_test_enable(desc.depth_stencil.stencil_test_enable);

        let color_blend_attachment = if desc.alpha_blend_enable {
            vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }
        } else {
            vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ZERO,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            }
        };

        let arr = [color_blend_attachment];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false).attachments(&arr);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = desc.outputs.color.iter().map(|f| f.to_vk_format()).collect::<Vec<vk::Format>>();

        //Dynamic rendering
        let mut dynamic_rendering_info = {
            let a = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(color_formats.as_slice());
            let b = if desc.outputs.depth.is_some() { a.depth_attachment_format(desc.outputs.depth.clone().unwrap().to_vk_format()) } else { a };

            let c = if desc.outputs.stencil.is_some() {
                b.stencil_attachment_format(desc.outputs.stencil.clone().unwrap().to_vk_format())
            } else {
                b
            };

            c
        };

        //Pipeline info
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut dynamic_rendering_info);

        let pipeline = unsafe { self.device.handle.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None).expect("Failed to create graphics pipeline")[0] };

        unsafe {
            for m in shader_modules {
                self.device.handle.destroy_shader_module(m, None);
            }
        }

        let raw_id = unsafe {
            (&mut *self.pipelines.get()).add(PipelineSlot {
                pipeline: pipeline,
                layout: pipeline_layout,
                bind_point: vk::PipelineBindPoint::GRAPHICS,
                push_constants_info: desc.push_constants,
            })
        };

        return Pipeline::Rasterization(raw_id);
    }

    pub(crate) fn create_compute_pipeline(&self, compute_pipeline_desc: &ComputePipelineDescription) -> Pipeline {
        let shader_module = self.create_shader_module(compute_pipeline_desc.shader_path);

        // pipeline layout
        let push_constant_ranges = [vk::PushConstantRange::default()
            .offset(compute_pipeline_desc.push_constants.offset)
            .size(compute_pipeline_desc.push_constants.size)
            .stage_flags(compute_pipeline_desc.push_constants.stage_flags.to_vk())];
        let layouts = [self.desc_layout];
        let layout_info = if compute_pipeline_desc.push_constants.size == 0 {
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts)
        } else {
            vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push_constant_ranges)
        };

        let pipeline_layout = unsafe { self.device.handle.create_pipeline_layout(&layout_info, None).expect("Failed to create pipeline layout") };

        let entry_point = std::ffi::CString::new("main").unwrap();

        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader_module).name(&entry_point);

        let pipeline_info = [vk::ComputePipelineCreateInfo::default().layout(pipeline_layout).stage(shader_stage_info)];

        let pipeline = unsafe { self.device.handle.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None).expect("Failed to create compute pipeline") }[0];

        unsafe {
            self.device.handle.destroy_shader_module(shader_module, None);
        }

        let raw_id = unsafe {
            (&mut *self.pipelines.get()).add(PipelineSlot {
                pipeline: pipeline,
                layout: pipeline_layout,
                bind_point: vk::PipelineBindPoint::COMPUTE,
                push_constants_info: compute_pipeline_desc.push_constants,
            })
        };

        return Pipeline::Compute(raw_id);
    }

    pub(crate) fn destroy_pipeline(&self, pipeline: Pipeline) {
        let slot = unsafe { (&mut *self.pipelines.get()).delete(pipeline.get_raw()) };

        unsafe {
            self.device.handle.destroy_pipeline_layout(slot.layout, None);
            self.device.handle.destroy_pipeline(slot.pipeline, None);
        }
    }
}

impl Drop for InnerPipelineManager {
    fn drop(&mut self) {
        let pipelines = unsafe { &mut (*self.pipelines.get()) };

        for page in pipelines.data.iter() {
            for (res, _) in page {
                res.map(|slot| unsafe {
                    self.device.handle.destroy_pipeline_layout(slot.layout, None);
                    self.device.handle.destroy_pipeline(slot.pipeline, None);
                });
            }
        }
    }
}
