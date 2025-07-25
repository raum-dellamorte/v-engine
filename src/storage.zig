const std = @import("std");
const vk = @import("vulkan");

const GraphicsContext = @import("gc.zig").GraphicsContext;

pub const StorageImage = struct {
    handle: vk.Image,
    device_memory: vk.DeviceMemory,
    image_layout: vk.ImageLayout = .general,

    const This = @This();

    pub fn init(gc: *const GraphicsContext, extent: vk.Extent2D, pool: vk.CommandPool) !This {
        var self: This = undefined;
        const format = vk.Format.r8g8b8a8_unorm;
        const format_prop = gc.instance.getPhysicalDeviceFormatProperties(gc.pdev, format);
        std.debug.assert(
            format_prop.optimal_tiling_features.contains(.{ .storage_image_bit = true }),
        );

        const image_create_info = vk.ImageCreateInfo{
            .image_type = .@"2d",
            .format = format,
            .extent = vk.Extent3D{
                .width = extent.width,
                .height = extent.height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .initial_layout = .undefined,
            .usage = vk.ImageUsageFlags{
                .storage_bit = true,
                .sampled_bit = true,
            },
            .sharing_mode = .exclusive,
        };

        self.handle = try gc.dev.createImage(&image_create_info, null);

        const mem_reqs = gc.dev.getImageMemoryRequirements(self.handle);
        const mem_alloc_info = vk.MemoryAllocateInfo{
            .allocation_size = mem_reqs.size,
            .memory_type_index = try gc.findMemoryTypeIndex(
                mem_reqs.memory_type_bits,
                .{ .device_local_bit = true },
            ),
        };

        self.device_memory = try gc.dev.allocateMemory(&mem_alloc_info, null);
        try gc.dev.bindImageMemory(self.handle, self.device_memory, 0);

        const layout_cmd_create_info = vk.CommandBufferAllocateInfo{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        };
        var layout_cmds: [1]vk.CommandBuffer = undefined;
        try gc.dev.allocateCommandBuffers(&layout_cmd_create_info, &layout_cmds);

        const layout_cmd_begin_info = vk.CommandBufferBeginInfo{};
        try gc.dev.beginCommandBuffer(layout_cmds[0], &layout_cmd_begin_info);

        const image_barrier = vk.ImageMemoryBarrier{
            .image = self.handle,
            .old_layout = .undefined,
            .new_layout = self.image_layout,
            .src_access_mask = .{},
            .dst_access_mask = .{},
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };
        _ = image_barrier; // autofix

        return self;
    }
};

