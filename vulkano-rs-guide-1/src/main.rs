//Code based on the official vulkano guide

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

fn main() {
    // Initialization
    // The instance maps vulkano to the local vulkan instalation
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
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
        .position(|(_, q)| q.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("couldn't find a graphical queue family") as u32;

    // The logic device is the software interface that represents the application's interaction with
    // the physical GPU
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
        .expect("failed to create device");

    // Iterators are lazy so the obtained queue needs to be initialized
    let queue = queues.next().unwrap();

    // A memory allocator is necessary before creating buffers in memory
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Example operation
    // Create a source buffer
    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        source_content,
    )
        .expect("failed to create source buffer");

    // Create a destination buffer
    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        destination_content,
    )
        .expect("failed to create destination buffer");

    // Like data buffers, command buffers need a dedicated memory allocator
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // Create a primary command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();

    // Build the actual command buffer
    let command_buffer = builder.build().unwrap();

    // Start the execution
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush() // same as signal fence, and then flush
        .unwrap();
    // Wait for the GPU to finish
    future.wait(None).unwrap();

    // The operation has succeeded
    let src_content = source.read().unwrap();
    let destination_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);

    println!("Everything succeeded!");
}
