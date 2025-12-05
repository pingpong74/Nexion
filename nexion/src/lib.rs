pub(crate) mod backend;

pub mod core;
pub mod definations;
pub mod taskgraph;
pub mod utils;

pub use core::{commands::*, device::*, gpu_resources::*, instance::*, pipelines::*, swapchain::*};
pub use definations::{commands::*, core::*, gpu_resources::*, pipelines::*};
pub use taskgraph::{definations::*, task_graph::*};

pub use bytemuck;
pub use memoffset;

// For copying the nexion.slang file to your directory.

pub fn add_shader_directory(paths: &str) {}

//Macros here
//
// Vertex macro

#[macro_export]
macro_rules! vertex {
    (
        $name:ident {
            input_rate: $rate:ident,
            $( $field:ident : $ty:ty ),* $(,)?
        }
    ) => {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name {
            $( pub $field: $ty, )*
        }

        impl $name {
            pub fn vertex_input_description() -> $crate::VertexInputDescription {
                use std::mem;
                let mut location = 0u32;

                let mut attributes = Vec::new();
                $(
                    attributes.push($crate::VertexAttribute {
                        location,
                        binding: 0,
                        format: <$ty as $crate::VertexFormat>::FORMAT,
                        offset: memoffset::offset_of!($name, $field) as u32,
                    });
                    location += 1;
                )*

                $crate::VertexInputDescription {
                    bindings: vec![
                        $crate::VertexBinding {
                            binding: 0,
                            stride: mem::size_of::<Self>() as u32,
                            input_rate: $crate::VertexInputRate::$rate,
                        }
                    ],
                    attributes,
                }
            }
        }
    };
}
