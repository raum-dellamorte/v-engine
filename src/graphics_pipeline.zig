const std = @import("std");
const vk = @import("vulkan");

const GraphicsContext = @import("gc.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const StorageImage = @import("images.zig").StorageImage;
const DepthStencil = @import("images.zig").DepthStencil;

const vert_spv align(@alignOf(u32)) = @embedFile("vertex_shader").*;
const frag_spv align(@alignOf(u32)) = @embedFile("fragment_shader").*;

pub const GraphicsPipeline = struct {
    handle: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    descriptor_set: vk.DescriptorSet,

    command_pool: vk.CommandPool,
    command_buffers: []vk.CommandBuffer,
    wait_fences: []vk.Fence,
    render_pass: vk.RenderPass,
    framebuffers: []vk.Framebuffer,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        gc: *const GraphicsContext,
        len: usize,
        allocator: std.mem.Allocator,
        swapchain: Swapchain,
        comman_pool: vk.CommandPool,
        descriptor_pool: vk.DescriptorPool,
        storage_image: StorageImage,
        depth_stencil: DepthStencil,
    ) !Self {
        var self: Self = undefined;
        self.allocator = allocator;
        self.command_pool = comman_pool;

        const fence_create_info = vk.FenceCreateInfo{
            .flags = .{ .signaled_bit = true },
        };
        self.wait_fences = try self.allocator.alloc(vk.Fence, len);
        for (0..len) |i| {
            self.wait_fences[i] = try gc.dev.createFence(&fence_create_info, null);
        }
        try self.createRenderPass(gc, swapchain);
        try self.createFramebuffers(gc, swapchain, depth_stencil);
        try self.initPipeline(gc, descriptor_pool, storage_image);
        try self.buildCommandBuffers(gc, len, swapchain.extent, storage_image);

        return self;
    }

    pub fn deinit(self: Self, gc: *const GraphicsContext) void {
        gc.dev.freeCommandBuffers(
            self.command_pool,
            @truncate(self.command_buffers.len),
            self.command_buffers.ptr,
        );
        self.allocator.free(self.command_buffers);
        self.deinitPipeline(gc);
        for (self.framebuffers) |fb| gc.dev.destroyFramebuffer(fb, null);
        self.allocator.free(self.framebuffers);
        gc.dev.destroyRenderPass(self.render_pass, null);
        for (self.wait_fences) |fence| gc.dev.destroyFence(fence, null);
        self.allocator.free(self.wait_fences);
    }

    pub fn resize(
        self: *Self,
        gc: *const GraphicsContext,
        len: usize,
        swapchain: Swapchain,
        storage_image: StorageImage,
        depth_stencil: DepthStencil,
    ) !void {
        for (self.framebuffers) |fb| gc.dev.destroyFramebuffer(fb, null);
        self.allocator.free(self.framebuffers);
        try self.createFramebuffers(gc, swapchain, depth_stencil);

        gc.dev.freeCommandBuffers(
            self.command_pool,
            @truncate(self.command_buffers.len),
            self.command_buffers.ptr,
        );
        self.allocator.free(self.command_buffers);
        try self.buildCommandBuffers(gc, len, swapchain.extent, storage_image);

        for (self.wait_fences) |fence| gc.dev.destroyFence(fence, null);
        self.allocator.free(self.wait_fences);

        const fence_create_info = vk.FenceCreateInfo{
            .flags = .{ .signaled_bit = true },
        };
        self.wait_fences = try self.allocator.alloc(vk.Fence, len);
        for (0..len) |i| {
            self.wait_fences[i] = try gc.dev.createFence(&fence_create_info, null);
        }
    }

    fn buildCommandBuffers(
        self: *Self,
        gc: *const GraphicsContext,
        len: usize,
        extent: vk.Extent2D,
        storage_image: StorageImage,
    ) !void {
        self.command_buffers = try self.allocator.alloc(vk.CommandBuffer, len);

        const draw_cmdbufs_alloc_info = vk.CommandBufferAllocateInfo{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = @intCast(self.command_buffers.len),
        };
        _ = try gc.dev.allocateCommandBuffers(&draw_cmdbufs_alloc_info, @ptrCast(self.command_buffers));
        const different_queue_family = gc.graphics_queue.family != gc.compute_queue.family;
        const cmd_buf_info = vk.CommandBufferBeginInfo{};

        const clear_value = [_]vk.ClearValue{
            .{ .color = .{ .float_32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } },
            .{ .depth_stencil = .{ .depth = 1.0, .stencil = 0 } },
        };

        var render_pass_begin_info = vk.RenderPassBeginInfo{
            .render_pass = self.render_pass,
            .framebuffer = undefined,
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = extent },
            .clear_value_count = clear_value.len,
            .p_clear_values = @ptrCast(&clear_value),
        };

        for (0..len) |i| {
            render_pass_begin_info.framebuffer = self.framebuffers[i];
            try gc.dev.beginCommandBuffer(self.command_buffers[i], &cmd_buf_info);

            var image_memory_barrier = vk.ImageMemoryBarrier{
                .src_access_mask = .{},
                .dst_access_mask = .{},
                .old_layout = .general,
                .new_layout = .general,
                .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .image = storage_image.handle,
                .subresource_range = vk.ImageSubresourceRange{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            };

            if (different_queue_family) {
                image_memory_barrier.dst_access_mask = .{ .shader_read_bit = true };
                image_memory_barrier.src_queue_family_index = gc.compute_queue.family;
                image_memory_barrier.dst_queue_family_index = gc.graphics_queue.family;
                gc.dev.cmdPipelineBarrier(
                    self.command_buffers[i],
                    .{ .top_of_pipe_bit = true },
                    .{ .fragment_shader_bit = true },
                    .{},
                    0,
                    null,
                    0,
                    null,
                    1,
                    @ptrCast(&image_memory_barrier),
                );
            } else {
                image_memory_barrier.src_access_mask = .{ .shader_write_bit = true };
                image_memory_barrier.dst_access_mask = .{ .shader_read_bit = true };
                gc.dev.cmdPipelineBarrier(
                    self.command_buffers[i],
                    .{ .compute_shader_bit = true },
                    .{ .fragment_shader_bit = true },
                    .{},
                    0,
                    null,
                    0,
                    null,
                    1,
                    @ptrCast(&image_memory_barrier),
                );
            }

            gc.dev.cmdBeginRenderPass(self.command_buffers[i], &render_pass_begin_info, .@"inline");

            const viewport = vk.Viewport{
                .x = 0,
                .y = 0,
                .width = @as(f32, @floatFromInt(extent.width)),
                .height = @as(f32, @floatFromInt(extent.height)),
                .min_depth = 0.0,
                .max_depth = 1.0,
            };
            gc.dev.cmdSetViewport(self.command_buffers[i], 0, 1, @ptrCast(&viewport));

            const scissor = vk.Rect2D{
                .offset = .{ .x = 0, .y = 0 },
                .extent = extent,
            };
            gc.dev.cmdSetScissor(self.command_buffers[i], 0, 1, @ptrCast(&scissor));

            gc.dev.cmdBindDescriptorSets(
                self.command_buffers[i],
                .graphics,
                self.pipeline_layout,
                0,
                1,
                @ptrCast(&self.descriptor_set),
                0,
                null,
            );
            gc.dev.cmdBindPipeline(self.command_buffers[i], .graphics, self.handle);
            gc.dev.cmdDraw(self.command_buffers[i], 3, 1, 0, 0);
            // draw UI
            gc.dev.cmdEndRenderPass(self.command_buffers[i]);

            if (different_queue_family) {
                image_memory_barrier.src_access_mask = .{ .shader_write_bit = true };
                image_memory_barrier.dst_access_mask = .{};
                image_memory_barrier.src_queue_family_index = gc.graphics_queue.family;
                image_memory_barrier.dst_queue_family_index = gc.compute_queue.family;
                gc.dev.cmdPipelineBarrier(
                    self.command_buffers[i],
                    .{ .fragment_shader_bit = true },
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

            try gc.dev.endCommandBuffer(self.command_buffers[i]);
        }
    }

    fn createRenderPass(self: *Self, gc: *const GraphicsContext, swapchain: Swapchain) !void {
        const color_attachment = vk.AttachmentDescription{
            .format = swapchain.surface_format.format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        };

        const depth_attachment = vk.AttachmentDescription{
            .format = gc.depth_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .depth_stencil_attachment_optimal,
        };

        const attachments = [_]vk.AttachmentDescription{
            color_attachment,
            depth_attachment,
        };

        const color_attachment_ref = vk.AttachmentReference{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        };

        const depth_attachment_ref = vk.AttachmentReference{
            .attachment = 1,
            .layout = .depth_stencil_attachment_optimal,
        };

        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&color_attachment_ref),
            .p_depth_stencil_attachment = @ptrCast(&depth_attachment_ref),
        };

        const dependecies = [_]vk.SubpassDependency{
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
                .dst_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
                .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
                .dst_access_mask = .{
                    .depth_stencil_attachment_read_bit = true,
                    .depth_stencil_attachment_write_bit = true,
                },
                .dependency_flags = .{},
            },
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{ .color_attachment_output_bit = true },
                .dst_stage_mask = .{ .color_attachment_output_bit = true },
                .src_access_mask = .{},
                .dst_access_mask = .{
                    .color_attachment_read_bit = true,
                    .color_attachment_write_bit = true,
                },
                .dependency_flags = .{},
            },
        };

        self.render_pass = try gc.dev.createRenderPass(&.{
            .attachment_count = attachments.len,
            .p_attachments = @ptrCast(&attachments),
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = dependecies.len,
            .p_dependencies = @ptrCast(&dependecies),
        }, null);
    }

    fn createFramebuffers(
        self: *Self,
        gc: *const GraphicsContext,
        swapchain: Swapchain,
        depth_stencil: DepthStencil,
    ) !void {
        self.framebuffers = try self.allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);

        for (self.framebuffers, 0..) |*fb, i| {
            const attachments = [_]vk.ImageView{
                swapchain.swap_images[i].view,
                depth_stencil.view,
            };
            fb.* = try gc.dev.createFramebuffer(&.{
                .render_pass = self.render_pass,
                .attachment_count = attachments.len,
                .p_attachments = @ptrCast(&attachments),
                .width = swapchain.extent.width,
                .height = swapchain.extent.height,
                .layers = 1,
            }, null);
        }
    }

    fn initPipeline(
        self: *Self,
        gc: *const GraphicsContext,
        descriptor_pool: vk.DescriptorPool,
        storage_image: StorageImage,
    ) !void {
        const set_layout_bidings = [_]vk.DescriptorSetLayoutBinding{
            .{
                .binding = 0,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = 1,
                .stage_flags = .{ .fragment_bit = true },
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
                .dst_set = self.descriptor_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .combined_image_sampler,
                .p_image_info = @ptrCast(&storage_image.descriptor),
                .p_buffer_info = @ptrCast(&[_]vk.DescriptorBufferInfo{}),
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
            .flags = .{},
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);

        const vert = try gc.dev.createShaderModule(&.{
            .code_size = vert_spv.len,
            .p_code = @ptrCast(&vert_spv),
        }, null);
        defer gc.dev.destroyShaderModule(vert, null);

        const frag = try gc.dev.createShaderModule(&.{
            .code_size = frag_spv.len,
            .p_code = @ptrCast(&frag_spv),
        }, null);
        defer gc.dev.destroyShaderModule(frag, null);

        const piasci = vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };

        const prsci = vk.PipelineRasterizationStateCreateInfo{
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .cull_mode = .{ .front_bit = true },
            .front_face = .counter_clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        };

        const pcbas = vk.PipelineColorBlendAttachmentState{
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        };

        const pcbsci = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&pcbas),
            .blend_constants = [_]f32{ 0, 0, 0, 0 },
        };

        const pdssci = vk.PipelineDepthStencilStateCreateInfo{
            .depth_test_enable = vk.FALSE,
            .depth_write_enable = vk.FALSE,
            .depth_compare_op = .less_or_equal,
            .depth_bounds_test_enable = vk.FALSE,
            .stencil_test_enable = vk.FALSE,
            .front = undefined,
            .back = .{
                .compare_op = .always,
                .fail_op = .keep,
                .pass_op = .keep,
                .depth_fail_op = .keep,
                .compare_mask = 0xFFFFFFFF,
                .write_mask = 0xFFFFFFFF,
                .reference = 0,
            },
            .min_depth_bounds = 0.0,
            .max_depth_bounds = 1.0,
        };

        const pvsci = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
            .scissor_count = 1,
            .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
        };

        const pmsci = vk.PipelineMultisampleStateCreateInfo{
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = vk.FALSE,
            .min_sample_shading = 1,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
        const pdsci = vk.PipelineDynamicStateCreateInfo{
            .flags = .{},
            .dynamic_state_count = dynstate.len,
            .p_dynamic_states = &dynstate,
        };

        const pssci = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .stage = .{ .vertex_bit = true },
                .module = vert,
                .p_name = "main",
            },
            .{
                .stage = .{ .fragment_bit = true },
                .module = frag,
                .p_name = "main",
            },
        };

        const gpci = vk.GraphicsPipelineCreateInfo{
            .stage_count = 2,
            .p_stages = &pssci,
            .p_vertex_input_state = &.{},
            .p_input_assembly_state = &piasci,
            .p_tessellation_state = null,
            .p_viewport_state = &pvsci,
            .p_rasterization_state = &prsci,
            .p_multisample_state = &pmsci,
            .p_depth_stencil_state = &pdssci,
            .p_color_blend_state = &pcbsci,
            .p_dynamic_state = &pdsci,
            .layout = self.pipeline_layout,
            .render_pass = self.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        _ = try gc.dev.createGraphicsPipelines(
            .null_handle,
            1,
            @ptrCast(&gpci),
            null,
            @ptrCast(&self.handle),
        );
    }

    fn deinitPipeline(self: *const Self, gc: *const GraphicsContext) void {
        gc.dev.destroyPipeline(self.handle, null);
        gc.dev.destroyPipelineLayout(self.pipeline_layout, null);
    }
};
