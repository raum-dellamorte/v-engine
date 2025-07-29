#version 450

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rgba8) uniform writeonly image2D resultImage;


layout (binding = 1) uniform UBO {
	vec4 color;
} ubo;

struct SceneObject
{
	vec4 objectProperties;
	vec3 diffuse;
	float specular;
	int id;
	int objectType;
};

layout (std140, binding = 2) buffer SceneObjects
{
	SceneObject sceneObjects[ ];
};

void main()
{
	ivec2 dim = imageSize(resultImage);
	vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;

    vec4 color = vec4(uv.x, uv.y, 0, 1);
			
	imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), color);
}
