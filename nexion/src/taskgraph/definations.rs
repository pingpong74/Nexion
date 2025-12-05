use crate::{BufferID, CommandRecorder, Device, ImageID, ImageLayout, ImageViewID, Swapchain};

#[derive(Clone, Copy, PartialEq)]
pub enum TaskAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskImageId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskBufferId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskImageViewId(pub usize);

#[derive(Clone, Copy, PartialEq)]
pub struct TaskImageAccess {
    pub id: TaskImageId,
    pub layout: ImageLayout,
    pub access: TaskAccess,
}

#[derive(Clone, Copy, PartialEq)]
pub struct TaskBufferAccess {
    pub id: TaskBufferId,
    pub access: TaskAccess,
}

#[derive(Clone, Copy, PartialEq)]
pub struct TaskImageViewAccess {
    pub id: TaskImageViewId,
    pub access: TaskAccess,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TaskResource {
    Image(TaskImageAccess),
    Buffer(TaskBufferAccess),
    ImageView(TaskImageViewAccess),
}

impl TaskResource {
    pub(crate) fn same_resource(a: &TaskResource, b: &TaskResource) -> bool {
        match (a, b) {
            (TaskResource::Image(_), TaskResource::Image(_)) => true,
            (TaskResource::Buffer(_), TaskResource::Buffer(_)) => true,
            (TaskResource::ImageView(_), TaskResource::ImageView(_)) => true,
            _ => false,
        }
    }

    pub(crate) fn get_access(&self) -> TaskAccess {
        match self {
            TaskResource::Image(img) => img.access,
            TaskResource::Buffer(buffer) => buffer.access,
            TaskResource::ImageView(img_view) => img_view.access,
        }
    }
}

pub struct TaskGraphInterface {
    pub recorder: CommandRecorder,
    images: &'static Vec<ImageID>,
    buffer: &'static Vec<BufferID>,
    image_views: &'static Vec<ImageViewID>,
}

pub struct Task {
    pub resources: Vec<TaskResource>,
    pub recorded_func: Box<dyn Fn(&TaskGraphInterface) + 'static>,
}

/// Information regarding the task graph
pub struct TaskGraphDescription {
    pub device: Device,
    pub swapchain: Option<Swapchain>,
}
