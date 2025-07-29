const vk = @import("vulkan");

const GraphicsContext = @import("gc.zig").GraphicsContext;
const StorageImage = @import("images.zig").StorageImage;
const Buffer = @import("buffer.zig").Buffer;

const comp_spv align(@alignOf(u32)) = @embedFile("compute_shader").*;

pub const ComputePipeline = struct {
    handle: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    pipeline_cache: vk.PipelineCache,
    descriptor_set: [1]vk.DescriptorSet,
    queue: vk.Queue,
    command_pool: vk.CommandPool,
    command_buffer: [1]vk.CommandBuffer,
    fence: vk.Fence,

    const Self = @This();

    pub fn init(
        gc: *const GraphicsContext,
        extent: vk.Extent2D,
        descriptor_pool: vk.DescriptorPool,
        storage_image: StorageImage,
        storage_buffer: Buffer,
        uniform_buffer: Buffer,
    ) !Self {
        var self: Self = undefined;
        self.queue = gc.dev.getDeviceQueue(gc.compute_queue.family, 0);

        const set_layout_bidings = [_]vk.DescriptorSetLayoutBinding{
            .{
                .binding = 0,
                .descriptor_type = .storage_image,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 1,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 2,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            },
        };

        const descriptor_layout = vk.DescriptorSetLayoutCreateInfo{
            .binding_count = set_layout_bidings.len,
            .p_bindings = @ptrCast(&set_layout_bidings),
        };

        const descriptor_set_layout = try gc.dev.createDescriptorSetLayout(&descriptor_layout, null);
        defer gc.dev.destroyDescriptorSetLayout(descriptor_set_layout, null);

        const alloc_info = vk.DescriptorSetAllocateInfo{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
        };
        _ = try gc.dev.allocateDescriptorSets(&alloc_info, @ptrCast(&self.descriptor_set));

        const descriptor_writes = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = self.descriptor_set[0],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_image,
                .p_image_info = @ptrCast(&[_]vk.DescriptorImageInfo{storage_image.descriptor}),
                .p_buffer_info = @ptrCast(&[_]vk.DescriptorBufferInfo{}),
                .p_texel_buffer_view = @ptrCast(&[_]vk.BufferView{}),
            },
            .{
                .dst_set = self.descriptor_set[0],
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .uniform_buffer,
                .p_image_info = @ptrCast(&[_]vk.DescriptorImageInfo{}),
                .p_buffer_info = @ptrCast(&[_]vk.DescriptorBufferInfo{uniform_buffer.descriptor}),
                .p_texel_buffer_view = @ptrCast(&[_]vk.BufferView{}),
            },
            .{
                .dst_set = self.descriptor_set[0],
                .dst_binding = 2,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = @ptrCast(&[_]vk.DescriptorImageInfo{}),
                .p_buffer_info = @ptrCast(&[_]vk.DescriptorBufferInfo{storage_buffer.descriptor}),
                .p_texel_buffer_view = @ptrCast(&[_]vk.BufferView{}),
            },
        };

        gc.dev.updateDescriptorSets(
            @intCast(descriptor_writes.len),
            @ptrCast(&descriptor_writes),
            0,
            null,
        );

        self.pipeline_layout = try gc.dev.createPipelineLayout(&.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);

        const comp = try gc.dev.createShaderModule(&.{
            .code_size = comp_spv.len,
            .p_code = @ptrCast(&comp_spv),
        }, null);
        defer gc.dev.destroyShaderModule(comp, null);

        const pipeline_cache_create_info = vk.PipelineCacheCreateInfo{};
        self.pipeline_cache = try gc.dev.createPipelineCache(&pipeline_cache_create_info, null);
        const compute_pipeline_create_info = vk.ComputePipelineCreateInfo{
            .stage = .{
                .stage = .{ .compute_bit = true },
                .module = comp,
                .p_name = "main",
            },
            .layout = self.pipeline_layout,
            .base_pipeline_index = -1,
        };
        _ = try gc.dev.createComputePipelines(
            self.pipeline_cache,
            1,
            @ptrCast(&compute_pipeline_create_info),
            null,
            @ptrCast(&self.handle),
        );

        const cmd_pool_create_info = vk.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = gc.compute_queue.family,
        };
        self.command_pool = try gc.dev.createCommandPool(&cmd_pool_create_info, null);

        const cmd_buf_alloc_info = vk.CommandBufferAllocateInfo{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        };
        _ = try gc.dev.allocateCommandBuffers(&cmd_buf_alloc_info, @ptrCast(&self.command_buffer));

        self.fence = try gc.dev.createFence(&.{}, null);

        try self.buildComputeCommandBuffer(gc, storage_image, extent);
        return self;
    }

    pub fn deinit(self: *const Self, gc: *const GraphicsContext) void {
        gc.dev.destroyPipeline(self.handle, null);
        gc.dev.destroyPipelineCache(self.pipeline_cache, null);
        gc.dev.destroyPipelineLayout(self.pipeline_layout, null);
        gc.dev.destroyCommandPool(self.command_pool, null);
        gc.dev.destroyFence(self.fence, null);
    }

    fn buildComputeCommandBuffer(
        self: *const Self,
        gc: *const GraphicsContext,
        storage_image: StorageImage,
        extent: vk.Extent2D,
    ) !void {
        const different_queue_families = gc.graphics_queue.family != gc.compute_queue.family;
        try gc.dev.beginCommandBuffer(self.command_buffer[0], &.{});

        var image_memory_barrier = vk.ImageMemoryBarrier{
            .image = storage_image.handle,
            .old_layout = .general,
            .new_layout = .general,
            .src_access_mask = .{},
            .dst_access_mask = .{},
            .subresource_range = vk.ImageSubresourceRange{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        };

        if (different_queue_families) {
            image_memory_barrier.dst_access_mask = .{ .shader_write_bit = true };
            image_memory_barrier.src_queue_family_index = gc.graphics_queue.family;
            image_memory_barrier.dst_queue_family_index = gc.compute_queue.family;
            gc.dev.cmdPipelineBarrier(
                self.command_buffer[0],
                .{ .top_of_pipe_bit = true },
                .{ .compute_shader_bit = true },
                .{},
                0,
                null,
                0,
                null,
                1,
                @ptrCast(&image_memory_barrier),
            );
        }

        gc.dev.cmdBindPipeline(self.command_buffer[0], .compute, self.handle);
        gc.dev.cmdBindDescriptorSets(
            self.command_buffer[0],
            .compute,
            self.pipeline_layout,
            0,
            1,
            @ptrCast(&self.descriptor_set),
            0,
            null,
        );

        gc.dev.cmdDispatch(self.command_buffer[0], extent.width / 16 + 1, extent.height / 16 + 1, 1);

        if (different_queue_families) {
            image_memory_barrier.src_access_mask = .{ .shader_write_bit = true };
            image_memory_barrier.dst_access_mask = .{};
            image_memory_barrier.src_queue_family_index = gc.compute_queue.family;
            image_memory_barrier.dst_queue_family_index = gc.graphics_queue.family;
            gc.dev.cmdPipelineBarrier(
                self.command_buffer[0],
                .{ .compute_shader_bit = true },
                .{ .bottom_of_pipe_bit = true },
                .{},
                0,
                null,
                0,
                null,
                1,
                @ptrCast(&image_memory_barrier),
            );
        }

        try gc.dev.endCommandBuffer(self.command_buffer[0]);
    }
};
