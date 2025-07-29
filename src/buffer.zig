const std = @import("std");
const vk = @import("vulkan");

const GraphicsContext = @import("gc.zig").GraphicsContext;

const UniformData = struct {
    color: [4]f32,
};

pub const Buffer = struct {
    handle: vk.Buffer,
    size: u64,
    alignment: u64,
    usage: vk.BufferUsageFlags,
    memory: vk.DeviceMemory,
    memory_flags: vk.MemoryPropertyFlags,
    descriptor: vk.DescriptorBufferInfo,
    mapped_data: ?*anyopaque = null,

    const Self = @This();

    pub fn init(
        gc: *const GraphicsContext,
        usage: vk.BufferUsageFlags,
        memory_flags: vk.MemoryPropertyFlags,
        size: u64,
        data: *anyopaque,
    ) !Self {
        var self: Self = undefined;

        const buffer_create_info = vk.BufferCreateInfo{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        };

        self.handle = try gc.dev.createBuffer(&buffer_create_info, null);

        const mem_reqs = gc.dev.getBufferMemoryRequirements(self.handle);
        var mem_alloc_info = vk.MemoryAllocateInfo{
            .allocation_size = mem_reqs.size,
            .memory_type_index = try gc.findMemoryTypeIndex(
                mem_reqs.memory_type_bits,
                memory_flags,
            ),
        };

        if (usage.contains(.{ .shader_device_address_bit = true })) {
            const alloc_flags_info = vk.MemoryAllocateFlagsInfoKHR{
                .flags = .{ .device_address_bit = true },
                .device_mask = 0,
            };
            mem_alloc_info.p_next = &alloc_flags_info;
        }

        self.memory = try gc.dev.allocateMemory(&mem_alloc_info, null);

        self.alignment = mem_reqs.alignment;
        self.size = mem_reqs.size;
        self.usage = usage;
        self.memory_flags = memory_flags;

        self.mapped_data = try gc.dev.mapMemory(self.memory, 0, self.size, .{});
        if (self.mapped_data) |ptr| {
            @memcpy(@as([*]u8, @ptrCast(ptr))[0..self.size], @as([*]u8, @ptrCast(data))[0..self.size]);
        }

        if (memory_flags.contains(.{ .host_coherent_bit = true })) {
            const mapped_range = [1]vk.MappedMemoryRange{.{
                .memory = self.memory,
                .offset = 0,
                .size = self.size,
            }};

            try gc.dev.flushMappedMemoryRanges(1, &mapped_range);
        }

        if (self.mapped_data) |_| {
            gc.dev.unmapMemory(self.memory);
            self.mapped_data = null;
        }

        self.descriptor = vk.DescriptorBufferInfo{
            .buffer = self.handle,
            .offset = 0,
            .range = self.size,
        };

        try gc.dev.bindBufferMemory(self.handle, self.memory, 0);

        return self;
    }

    pub fn deinit(self: *Self, gc: *const GraphicsContext) void {
        if (self.mapped_data) |_| {
            gc.dev.unmapMemory(self.memory);
            self.mapped_data = null;
        }

        gc.dev.destroyBuffer(self.handle, null);
        gc.dev.freeMemory(self.memory, null);
    }

    pub fn prepareStorageBuffer(gc: *const GraphicsContext) !Self {
        var data: [3]f32 = .{ 1.0, 2.0, 3.0 };
        const storage_buffer_size = data.len * @sizeOf(f32);

        return Buffer.init(
            gc,
            .{ .storage_buffer_bit = true },
            .{ .device_local_bit = true, .host_visible_bit = true },
            storage_buffer_size,
            @ptrCast(&data),
        );
    }

    pub fn prepareUniformBuffer(gc: *const GraphicsContext) !Self {
        var data: UniformData = .{ .color = .{ 0, 1, 0, 1 } };

        return Buffer.init(
            gc,
            .{ .uniform_buffer_bit = true },
            .{ .host_coherent_bit = true, .host_visible_bit = true },
            @sizeOf(UniformData),
            @ptrCast(&data),
        );
    }
};
