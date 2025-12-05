#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferID {
    pub(crate) id: u64,
}

impl BufferID {
    pub fn null() -> BufferID {
        return BufferID { id: u64::MAX };
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageID {
    pub(crate) id: u64,
}

impl ImageID {
    pub fn null() -> ImageID {
        return ImageID { id: u64::MAX };
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SamplerID {
    pub(crate) id: u64,
}

impl SamplerID {
    pub fn null() -> SamplerID {
        return SamplerID { id: u64::MAX };
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageViewID {
    pub(crate) id: u64,
}

impl ImageViewID {
    pub fn null() -> ImageViewID {
        return ImageViewID { id: u64::MAX };
    }
}
