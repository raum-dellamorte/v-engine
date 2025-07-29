const std = @import("std");
const vk = @import("vulkan");

const GraphicsContext = @import("gc.zig").GraphicsContext;

pub const StorageImage = struct {
    handle: vk.Image,
    device_memory: vk.DeviceMemory,
    image_layout: vk.ImageLayout,
    sampler: vk.Sampler,
    view: vk.ImageView,
    descriptor: vk.DescriptorImageInfo,

    const Self = @This();

    pub fn init(gc: *const GraphicsContext, extent: vk.Extent2D, pool: vk.CommandPool) !Self {
        var self: Self = undefined;
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

        self.image_layout = .general;

        const layout_cmd_create_info = vk.CommandBufferAllocateInfo{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        };
        var layout_cmds: [1]vk.CommandBuffer = undefined;
        try gc.dev.allocateCommandBuffers(&layout_cmd_create_info, &layout_cmds);

        const layout_cmd_begin_info = vk.CommandBufferBeginInfo{};
        try gc.dev.beginCommandBuffer(layout_cmds[0], &layout_cmd_begin_info);

        var gc_queue_family_indices: [2]u32 = .{ vk.QUEUE_FAMILY_IGNORED, vk.QUEUE_FAMILY_IGNORED };
        if (gc.graphics_queue.family != gc.compute_queue.family) {
            gc_queue_family_indices = .{
                gc.graphics_queue.family,
                gc.compute_queue.family,
            };
        }

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
            .src_queue_family_index = gc_queue_family_indices[0],
            .dst_queue_family_index = gc_queue_family_indices[1],
        };
        gc.dev.cmdPipelineBarrier(
            layout_cmds[0],
            .{ .bottom_of_pipe_bit = true },
            .{ .fragment_shader_bit = true },
            .{},
            0,
            null,
            0,
            null,
            1,
            @ptrCast(&image_barrier),
        );

        try gc.flushCommandBuffer(layout_cmds[0], gc.graphics_queue, pool, true);

        const sampler_create_info = vk.SamplerCreateInfo{
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = .clamp_to_border,
            .address_mode_v = .clamp_to_border,
            .address_mode_w = .clamp_to_border,
            .mip_lod_bias = 0.0,
            .anisotropy_enable = 0,
            .max_anisotropy = 1.0,
            .compare_op = .never,
            .compare_enable = 0,
            .min_lod = 0.0,
            .max_lod = 0.0,
            .border_color = .float_opaque_white,
            .unnormalized_coordinates = 0,
        };
        self.sampler = try gc.dev.createSampler(&sampler_create_info, null);

        const view_create_info = vk.ImageViewCreateInfo{
            .image = self.handle,
            .view_type = .@"2d",
            .format = format,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
        };
        self.view = try gc.dev.createImageView(&view_create_info, null);
        self.descriptor = vk.DescriptorImageInfo{
            .sampler = self.sampler,
            .image_view = self.view,
            .image_layout = self.image_layout,
        };

        return self;
    }

    pub fn deinit(self: Self, gc: *const GraphicsContext) void {
        gc.dev.destroyImageView(self.view, null);
        gc.dev.destroySampler(self.sampler, null);
        gc.dev.freeMemory(self.device_memory, null);
        gc.dev.destroyImage(self.handle, null);
    }
};

pub const DepthStencil = struct {
    image: vk.Image,
    view: vk.ImageView,
    memory: vk.DeviceMemory,

    const Self = @This();

    pub fn init(gc: *const GraphicsContext, extent: vk.Extent2D) !Self {
        var self: Self = undefined;
        const image_create_info = vk.ImageCreateInfo{
            .image_type = .@"2d",
            .format = gc.depth_format,
            .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{ .depth_stencil_attachment_bit = true },
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        };
        self.image = try gc.dev.createImage(&image_create_info, null);

        const mem_reqs = gc.dev.getImageMemoryRequirements(self.image);

        const mem_alloc_info = vk.MemoryAllocateInfo{
            .allocation_size = mem_reqs.size,
            .memory_type_index = try gc.findMemoryTypeIndex(
                mem_reqs.memory_type_bits,
                .{ .device_local_bit = true },
            ),
        };

        self.memory = try gc.dev.allocateMemory(&mem_alloc_info, null);
        try gc.dev.bindImageMemory(self.image, self.memory, 0);

        const view_create_info = vk.ImageViewCreateInfo{
            .image = self.image,
            .view_type = .@"2d",
            .format = gc.depth_format,
            .subresource_range = vk.ImageSubresourceRange{
                .aspect_mask = .{ .depth_bit = true, .stencil_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .components = vk.ComponentMapping{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
        };

        self.view = try gc.dev.createImageView(&view_create_info, null);

        return self;
    }

    pub fn deinit(self: *const Self, gc: *const GraphicsContext) void {
        gc.dev.destroyImageView(self.view, null);
        gc.dev.destroyImage(self.image, null);
        gc.dev.freeMemory(self.memory, null);
    }
};
