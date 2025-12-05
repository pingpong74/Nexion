use std::u64::MAX;

use ash::vk;
use vk_mem::*;

#[derive(Clone)]
pub(crate) struct BufferSlot {
    pub(crate) handle: vk::Buffer,
    pub(crate) address: vk::DeviceAddress,
    pub(crate) allocation: Allocation,
    pub(crate) alloc_info: AllocationInfo,
}

#[derive(Clone)]
pub(crate) struct ImageSlot {
    pub(crate) handle: vk::Image,
    pub(crate) allocation: Allocation,
    pub(crate) alloc_info: AllocationInfo,
    pub(crate) format: vk::Format,
}

#[derive(Clone)]
pub(crate) struct ImageViewSlot {
    pub(crate) handle: vk::ImageView,
    pub(crate) parent_image: vk::Image,
}

#[derive(Clone)]
pub(crate) struct SamplerSlot {
    pub(crate) handle: vk::Sampler,
}

const MASK: u64 = 0xFFFF;

fn encode(page: u64, index: u64, version: u64) -> u64 {
    return (page << 32) | (index << 16) | version;
}

// return -> (Page, index, version)
fn decode_as_usize(id: u64) -> (usize, usize, u64) {
    return (((id >> 32) & MASK) as usize, ((id >> 16) & MASK) as usize, (id & MASK));
}

// Be careful while changing!!!!!!!!
// its used in shader as well. (common.slang)
// both values MUST match!!
const PAGE_SIZE: usize = 10;

/// Assinging 16 bits to each of the numbers, paging, index and version
/// <---- Filler bits -----> 16 paging 16 index 16 version
///
/// Actual creation and destruction happens on a device, this just manages the ids

pub(crate) struct GpuResourcePool<Resource> {
    data: Vec<[(Option<Resource>, u64); PAGE_SIZE]>,
    free_indices: Vec<u64>,
    curr_page: usize,
    curr_index: usize,
}

impl<Resource> GpuResourcePool<Resource> {
    pub(crate) fn new() -> Self {
        return GpuResourcePool {
            data: vec![std::array::from_fn(|_| (None, 0))],
            free_indices: Vec::new(),
            curr_index: 0,
            curr_page: 0,
        };
    }

    pub(crate) fn add(&mut self, res: Resource) -> u64 {
        if self.free_indices.is_empty() {
            if self.curr_index == PAGE_SIZE {
                self.data.push(std::array::from_fn(|_| (None, 0)));
                self.curr_index = 0;
                self.curr_page += 1;
            }

            let id = encode(self.curr_page as u64, self.curr_index as u64, 0);

            self.data[self.curr_page][self.curr_index] = (Some(res), 0);

            self.curr_index += 1;

            return id;
        } else {
            let id = self.free_indices.pop().unwrap();

            let (page, index, version) = decode_as_usize(id);

            self.data[page][index] = (Some(res), version + 1);

            return encode(page as u64, index as u64, version + 1);
        }
    }

    pub(crate) fn delete(&mut self, id: u64) -> Resource {
        let (page, index, version) = decode_as_usize(id);

        let (res_opt, res_version) = &mut self.data[page][index];

        match res_opt.take() {
            Some(res) => {
                if *res_version == version {
                    self.data[page][index] = (None, version);
                    self.free_indices.push(id);

                    return res;
                } else {
                    panic!("Attempted to acess with invalid ID")
                }
            }
            None => {
                panic!("Attempted to acess with invalid ID")
            }
        }
    }

    pub(crate) fn get_ref(&self, id: u64) -> &Resource {
        let (page, index, version) = decode_as_usize(id);

        let (res_opt, res_version) = &self.data[page][index];

        match res_opt {
            Some(res) => {
                if *res_version == version {
                    return res;
                } else {
                    panic!("Attempted acess with invalid ID")
                }
            }
            None => {
                panic!("Attempted acess with invalid ID")
            }
        }
    }
}

/// Provides 4 resource types
/// Storage Buffer        -> binding 0
/// Sampled Image         -> binding 1
/// Storage image         -> binding 2
/// Sampler               -> binding 3
pub(crate) struct GpuBindlessDescriptorPool {
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) set: vk::DescriptorSet,
    pub(crate) layout: vk::DescriptorSetLayout,
}

impl GpuBindlessDescriptorPool {
    pub(crate) fn new(device: &ash::Device, max_buffers: u32, max_storage_images: u32, max_sampled_images: u32, max_samplers: u32) -> GpuBindlessDescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: max_buffers,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: max_storage_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: max_sampled_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: max_samplers,
            },
        ];

        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND | vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_create_info, None).expect("Failed to create bindless descriptor pool") };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(max_buffers)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(max_sampled_images)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(max_storage_images)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(max_samplers)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
        ];

        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .push_next(&mut binding_flags_info)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings);

        let bindless_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None).expect("Failed to create bindless descriptor set layout") };

        let variable_counts = [max_buffers];
        let mut variable_count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&variable_counts);

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&bindless_set_layout))
            .push_next(&mut variable_count_info);

        let bindless_set = unsafe { device.allocate_descriptor_sets(&alloc_info).expect("Failed to create bindless descriptor") }[0];

        return GpuBindlessDescriptorPool {
            pool: descriptor_pool,
            set: bindless_set,
            layout: bindless_set_layout,
        };
    }

    pub(crate) fn write_buffer(&self, device: &ash::Device, buffer: vk::Buffer, index: u32) {
        let buffer_info = [vk::DescriptorBufferInfo {
            buffer: buffer,
            offset: 0,
            range: MAX,
        }];

        let write_info = [vk::WriteDescriptorSet::default()
            .buffer_info(&buffer_info)
            .dst_set(self.set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)];

        unsafe {
            device.update_descriptor_sets(&write_info, &[]);
        }
    }

    pub(crate) fn write_sampled_image(&self, device: &ash::Device, image_view: vk::ImageView, index: u32) {
        let sampler_info = [vk::DescriptorImageInfo {
            image_view: image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        }];

        let write_info = [vk::WriteDescriptorSet::default()
            .image_info(&sampler_info)
            .dst_set(self.set)
            .dst_binding(1)
            .dst_array_element(index)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)];

        let copy_sets = [];

        unsafe {
            device.update_descriptor_sets(&write_info, &copy_sets);
        }
    }

    pub(crate) fn write_storage_image(&self, device: &ash::Device, image_view: vk::ImageView, index: u32) {
        let sampler_info = [vk::DescriptorImageInfo {
            image_view: image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        }];

        let write_info = [vk::WriteDescriptorSet::default()
            .image_info(&sampler_info)
            .dst_set(self.set)
            .dst_binding(2)
            .dst_array_element(index)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)];

        let copy_sets = [];

        unsafe {
            device.update_descriptor_sets(&write_info, &copy_sets);
        }
    }

    pub(crate) fn write_sampler(&self, device: &ash::Device, sampler: vk::Sampler, index: u32) {
        let sampler_info = [vk::DescriptorImageInfo {
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
            sampler: sampler,
        }];

        let write_info = [vk::WriteDescriptorSet::default()
            .image_info(&sampler_info)
            .dst_set(self.set)
            .dst_binding(3)
            .dst_array_element(index)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)];

        let copy_sets = [];

        unsafe {
            device.update_descriptor_sets(&write_info, &copy_sets);
        }
    }

    pub(crate) fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
