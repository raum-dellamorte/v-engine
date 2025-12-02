const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vulkan");

const GraphicsContext = @import("gc.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const StorageImage = @import("images.zig").StorageImage;
const DepthStencil = @import("images.zig").DepthStencil;
const Buffer = @import("buffer.zig").Buffer;
const ComputePipeline = @import("compute_pipeline.zig").ComputePipeline;
const GraphicsPipeline = @import("graphics_pipeline.zig").GraphicsPipeline;

const APP_NAME = "V-Engine";
const WIDTH = 1080;
const HEIGHT = 720;

pub const App = struct {
    window: *glfw.Window,
    extent: vk.Extent2D,

    gc: GraphicsContext,
    swapchain: Swapchain,
    command_pool: vk.CommandPool,
    descriptor_pool: vk.DescriptorPool,

    depth_stencil: DepthStencil,
    storage_image: StorageImage,
    storage_buffer: Buffer,
    uniform_buffer: Buffer,

    compute_pipeline: ComputePipeline,
    graphics_pipeline: GraphicsPipeline,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !App {
        var self: App = undefined;
        try self.initWindow();
        self.allocator = allocator;
        self.updateExtent();
        self.gc = try GraphicsContext.init(self.allocator, APP_NAME, self.window);
        self.swapchain = try Swapchain.init(&self.gc, self.allocator, self.extent);
        self.command_pool = try self.gc.dev.createCommandPool(&.{
            .queue_family_index = self.gc.graphics_queue.family,
        }, null);
        try self.setupDescriptorPool();
        self.depth_stencil = try DepthStencil.init(&self.gc, self.extent);
        self.storage_image = try StorageImage.init(&self.gc, self.extent, self.command_pool);
        self.storage_buffer = try Buffer.prepareStorageBuffer(&self.gc);
        self.uniform_buffer = try Buffer.prepareUniformBuffer(&self.gc);
        self.compute_pipeline = try ComputePipeline.init(
            &self.gc,
            self.extent,
            self.descriptor_pool,
            self.storage_image,
            self.storage_buffer,
            self.uniform_buffer,
        );
        self.graphics_pipeline = try GraphicsPipeline.init(
            &self.gc,
            self.swapchain.swap_images.len,
            self.allocator,
            self.swapchain,
            self.command_pool,
            self.descriptor_pool,
            self.storage_image,
            self.depth_stencil,
        );

        return self;
    }

    pub fn deinit(self: *App) void {
        self.graphics_pipeline.deinit(&self.gc);
        self.compute_pipeline.deinit(&self.gc);
        self.uniform_buffer.deinit(&self.gc);
        self.storage_buffer.deinit(&self.gc);
        self.storage_image.deinit(&self.gc);
        self.depth_stencil.deinit(&self.gc);
        self.gc.dev.destroyDescriptorPool(self.descriptor_pool, null);
        self.gc.dev.destroyCommandPool(self.command_pool, null);
        self.swapchain.deinit();
        self.gc.deinit();
        self.deinitWindow();
    }

    pub fn run(self: *App) !void {
        while (!glfw.windowShouldClose(self.window)) {
            // process inputs
            if (glfw.getKey(self.window, glfw.KeyEscape) == glfw.Press) {
                glfw.setWindowShouldClose(self.window, true);
            }

            // update

            const present_state = try self.draw();
            if (present_state == .suboptimal) {
                try self.resize();
            }

            glfw.pollEvents();
        }

        // Needed to safely free the command buffers
        try self.gc.dev.queueWaitIdle(self.gc.graphics_queue.handle);
    }

    fn draw(self: *App) !Swapchain.PresentState {
        const compute_sumbit_into = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.compute_pipeline.command_buffer[0]),
        };

        try self.gc.dev.queueSubmit(
            self.compute_pipeline.queue,
            1,
            @ptrCast(&compute_sumbit_into),
            self.compute_pipeline.fence,
        );
        _ = try self.gc.dev.waitForFences(
            1,
            @ptrCast(&self.compute_pipeline.fence),
            .true,
            std.math.maxInt(u64),
        );
        try self.gc.dev.resetFences(1, @ptrCast(&self.compute_pipeline.fence));

        return self.swapchain.present(
            self.graphics_pipeline.command_buffers[self.swapchain.image_index],
        );
    }

    fn resize(self: *App) !void {
        self.updateExtent();

        while (self.extent.width == 0 or self.extent.height == 0) {
            self.updateExtent();
            glfw.waitEvents();
        }

        try self.gc.dev.deviceWaitIdle();
        try self.gc.dev.queueWaitIdle(self.gc.graphics_queue.handle);
        try self.gc.dev.queueWaitIdle(self.gc.compute_queue.handle);

        try self.swapchain.recreate(self.extent);

        self.depth_stencil.deinit(&self.gc);
        self.depth_stencil = try DepthStencil.init(&self.gc, self.extent);

        self.storage_image.deinit(&self.gc);
        self.storage_image = try StorageImage.init(&self.gc, self.extent, self.command_pool);

        self.graphics_pipeline.deinit(&self.gc);
        self.compute_pipeline.deinit(&self.gc);

        self.gc.dev.destroyDescriptorPool(self.descriptor_pool, null);
        try self.setupDescriptorPool();

        self.compute_pipeline = try ComputePipeline.init(
            &self.gc,
            self.extent,
            self.descriptor_pool,
            self.storage_image,
            self.storage_buffer,
            self.uniform_buffer,
        );

        self.graphics_pipeline = try GraphicsPipeline.init(
            &self.gc,
            self.swapchain.swap_images.len,
            self.allocator,
            self.swapchain,
            self.command_pool,
            self.descriptor_pool,
            self.storage_image,
            self.depth_stencil,
        );

        try self.gc.dev.deviceWaitIdle();
    }

    fn initWindow(self: *App) !void {
        try glfw.init();
        glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);
        self.window = try glfw.createWindow(WIDTH, HEIGHT, APP_NAME, null, null);
    }

    fn deinitWindow(self: *App) void {
        glfw.destroyWindow(self.window);
        glfw.terminate();
    }

    fn updateExtent(self: *App) void {
        var width: c_int = undefined;
        var height: c_int = undefined;
        glfw.getFramebufferSize(self.window, &width, &height);
        self.extent = vk.Extent2D{
            .width = @intCast(width),
            .height = @intCast(height),
        };
    }

    fn setupDescriptorPool(self: *App) !void {
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .uniform_buffer, .descriptor_count = 2 },
            .{ .type = .combined_image_sampler, .descriptor_count = 4 },
            .{ .type = .storage_image, .descriptor_count = 1 },
            .{ .type = .storage_buffer, .descriptor_count = 2 },
        };

        const pool_create_info = vk.DescriptorPoolCreateInfo{
            .max_sets = 3,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = @ptrCast(&pool_sizes),
        };

        self.descriptor_pool = try self.gc.dev.createDescriptorPool(&pool_create_info, null);
    }
};
