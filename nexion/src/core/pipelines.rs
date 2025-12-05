use ash::vk;
use std::sync::Arc;

use crate::{
    ComputePipelineDescription, RasterizationPipelineDescription, ShaderStages,
    backend::pipelines::{InnerComputePipeline, InnerPipelineManager, InnerRasterizationPipeline},
};

#[derive(Clone)]
pub struct PipelineManager {
    pub(crate) inner: Arc<InnerPipelineManager>,
}

impl PipelineManager {
    pub fn create_rasterization_pipeline(&self, raster_pipeline_desc: &RasterizationPipelineDescription) -> RasterizationPipeline {
        let (pipeline, layout) = self.inner.create_raster_pipeline_data(raster_pipeline_desc);

        return RasterizationPipeline {
            inner: Arc::new(InnerRasterizationPipeline {
                handle: pipeline,
                layout: layout,
                desc: raster_pipeline_desc.clone(),
                manager: self.inner.clone(),
            }),
        };
    }

    pub fn create_compute_pipeline(&self, compute_pipeline_desc: &ComputePipelineDescription) -> ComputePipeline {
        let (pipeline, layout) = self.inner.create_compute_pipeline(compute_pipeline_desc);
        return ComputePipeline {
            inner: Arc::new(InnerComputePipeline {
                handle: pipeline,
                layout: layout,
                desc: compute_pipeline_desc.clone(),
                manager: self.inner.clone(),
            }),
        };
    }
}

pub struct RasterizationPipeline {
    pub(crate) inner: Arc<InnerRasterizationPipeline>,
}

pub struct ComputePipeline {
    pub(crate) inner: Arc<InnerComputePipeline>,
}

pub trait Pipeline {
    fn get_push_const_shader_stage(&self) -> ShaderStages;
    fn get_layout(&self) -> vk::PipelineLayout;
    fn get_handle(&self) -> vk::Pipeline;
    fn get_bind_point(&self) -> vk::PipelineBindPoint;
}

impl Pipeline for RasterizationPipeline {
    fn get_push_const_shader_stage(&self) -> ShaderStages {
        return self.inner.desc.push_constants.stage_flags;
    }
    fn get_handle(&self) -> vk::Pipeline {
        return self.inner.handle;
    }
    fn get_bind_point(&self) -> vk::PipelineBindPoint {
        return vk::PipelineBindPoint::GRAPHICS;
    }
    fn get_layout(&self) -> vk::PipelineLayout {
        return self.inner.layout;
    }
}

impl Pipeline for ComputePipeline {
    fn get_push_const_shader_stage(&self) -> ShaderStages {
        return self.inner.desc.push_constants.stage_flags;
    }
    fn get_handle(&self) -> vk::Pipeline {
        return self.inner.handle;
    }
    fn get_bind_point(&self) -> vk::PipelineBindPoint {
        return vk::PipelineBindPoint::COMPUTE;
    }
    fn get_layout(&self) -> vk::PipelineLayout {
        return self.inner.layout;
    }
}
