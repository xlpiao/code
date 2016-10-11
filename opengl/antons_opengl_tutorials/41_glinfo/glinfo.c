#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <string.h>


static void
query_ATI_meminfo(void)
{
    // #ifdef GL_ATI_meminfo
    int mem[4];
    printf("Memory info (GL_ATI_meminfo):\n");
    glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, i);
    printf("    VBO free memory - total: %u MB, largest block: %u MB\n", mem[0] / 1024, mem[1] / 1024);
    printf("    VBO free aux. memory - total: %u MB, largest block: %u MB\n", mem[2] / 1024, mem[3] / 1024);
    glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, i);
    printf("    Texture free memory - total: %u MB, largest block: %u MB\n", mem[0] / 1024, mem[1] / 1024);
    printf("    Texture free aux. memory - total: %u MB, largest block: %u MB\n", mem[2] / 1024, mem[3] / 1024);
    glGetIntegerv(GL_RENDERBUFFER_FREE_MEMORY_ATI, i);
    printf("    Renderbuffer free memory - total: %u MB, largest block: %u MB\n", mem[0] / 1024, mem[1] / 1024);
    printf("    Renderbuffer free aux. memory - total: %u MB, largest block: %u MB\n", mem[2] / 1024, mem[3] / 1024);
    // #endif
}
static void
query_NVX_gpu_memory_info(void)
{
    // #ifdef GL_NVX_gpu_memory_info
    int mem;
    printf("Memory info (GL_NVX_gpu_memory_info):\n");
    glGetIntegerv(GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX, &mem);
    printf("    Dedicated video memory: %u MB\n", mem / 1024);
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &mem);
    printf("    Total available memory: %u MB\n", mem / 1024);
    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &mem);
    printf("    Currently available dedicated video memory: %u MB\n", mem / 1024);
    // #endif
}
void
print_gpu_memory_info(const GLubyte * glExtensions)
{
    if (strstr((const char *)glExtensions, "GL_ATI_meminfo"))
        query_ATI_meminfo();
    else if (strstr((const char *)glExtensions, "GL_NVX_gpu_memory_info"))
        query_NVX_gpu_memory_info();
    else if (strstr((const char *)glExtensions, "ATI Corporation"))
        query_ATI_meminfo();
    else if (strstr((const char *)glExtensions, "NVIDIA Corporation"))
        query_NVX_gpu_memory_info();
    else if (strstr((const char *)glExtensions, "Intel Inc."))
        query_ATI_meminfo();
    else
        printf("Cannot query GPU MEMORY\n");
}


int main (void) {
	GLFWwindow* window = NULL;
	const GLubyte* renderer;
	const GLubyte* version;
	const GLubyte* vendor;

	if (!glfwInit ()) {
		fprintf (stderr, "ERROR: could not start GLFW3\n");
		return 1;
	} 

	window = glfwCreateWindow (640, 480, "GL INFO", NULL, NULL);
	glfwMakeContextCurrent (window);
	glewExperimental = GL_TRUE;
	glewInit ();

	renderer = glGetString (GL_RENDERER);
	printf ("Renderer: %s\n", renderer);

	version = glGetString (GL_VERSION);
	printf ("OpenGL version supported %s\n", version);

    vendor = glGetString(GL_VENDOR); 
	printf ("Vendor: %s\n", vendor);

    print_gpu_memory_info(vendor);

	return 0;
}
