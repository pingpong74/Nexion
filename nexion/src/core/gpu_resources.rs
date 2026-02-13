use std::u64;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferId {
    pub(crate) id: u64,
}

impl BufferId {
    pub const fn null() -> BufferId {
        return BufferId { id: u64::MAX };
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageId {
    pub(crate) id: u64,
}

impl ImageId {
    pub const fn null() -> ImageId {
        return ImageId { id: u64::MAX };
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SamplerId {
    pub(crate) id: u64,
}

impl SamplerId {
    pub const fn null() -> SamplerId {
        return SamplerId { id: u64::MAX };
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageViewId {
    pub(crate) id: u64,
}

impl ImageViewId {
    pub const fn null() -> ImageViewId {
        return ImageViewId { id: u64::MAX };
    }
}

// pipelines

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Pipeline {
    Rasterization(u64),
    Compute(u64),
}

impl Pipeline {
    pub const fn null() -> Pipeline {
        return Pipeline::Rasterization(u64::MAX);
    }

    pub(crate) fn get_raw(&self) -> u64 {
        return match self {
            Pipeline::Compute(id) => *id,
            Pipeline::Rasterization(id) => *id,
        };
    }
}
