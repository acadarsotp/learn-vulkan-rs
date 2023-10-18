//Code based on the official vulkano guide

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

fn main() {
    // Initialization
    // The instance maps vulkano to the local vulkan instalation
    let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

    // Select Nvidia GPU
    // The physical device is the graphics card to be used
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .skip(1)
        .next()
        .expect("no devices available");


    // Device creation

    // In a GPU queues are equivalent to CPU threads, GPUs have thread families that support different
    // operations
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::COMPUTE)
        })
        .expect("couldn't find a compute queue family") as u32;

    // The logic device is the software interface that represents the application's interaction with
    // the physical GPU
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    )
        .expect("failed to create device");

    // Iterators are lazy so the obtained queue needs to be initialized
    let queue = queues.next().unwrap();

    // A memory allocator is necessary before creating buffers in memory
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Create a data buffer
    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        data_iter,
    )
        .expect("failed to create buffer");

    // Compute pipelines
    // We are going to multiply the 65536 values on the data buffer by 12
    // GLSL shader to program the actual parallel computing
    /*
        #version 460 -> The GLSL version to be used

                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in; -> Defining the
                    local size of the work groups, 1024 work groups * local size of 64 = 65536
                    x, y, z for convenience if working with 2d or 3d data structure

                layout(set = 0, binding = 0) buffer Data {
                    uint data[];
                } buf; -> Creating a slot for a descriptor set, descriptors describe the resources
                          that shaders will use during execution (data buffer in this case),
                          provide a way to bind resources to shaders and specify how they can be
                          accessed by the shaders


                void main() { -> shader entry point
                    uint idx = gl_GlobalInvocationID.x; -> represents the indices of the buffer
                                                            0..65536
                    buf.data[idx] *= 12; -> multiply each index by 12
                }
     */
    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
                #version 460

                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                layout(set = 0, binding = 0) buffer Data {
                    uint data[];
                } buf;

                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] *= 12;
                }
            "
        }
    }

    let shader = cs::load(device.clone()).expect("failed to create shader module");

    // Create a computer pipeline object from the shader
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
        .expect("failed to create compute pipeline");

    // Just like buffers and command buffers, descriptor sets need allocators
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    // Before creating the descriptor set, the layout its targeting is needed
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();

    // Create the descriptor set
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
    )
        .unwrap();

    // Create command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    let work_group_counts = [1024, 1, 1];

    // Bind the pipeline and descriptor sets
    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .dispatch(work_group_counts)
        .unwrap();

    // Build the command buffer
    let command_buffer = command_buffer_builder.build().unwrap();

    // Start execution
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // Wait for GPU to finish
    future.wait(None).unwrap();

    // The operation has succeeded
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");
}
