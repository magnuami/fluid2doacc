#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include <GL/glew.h>
#include "gl_helper.h"
#include <string.h>
#include <openacc.h>

extern void initFluid(struct Configuration *config, int dimX, int dimY);
extern void freeFluid(struct Configuration *config);
extern void solveFluid(struct Configuration *config);
extern void renderFluid(struct Configuration *config, float4 *d_output);

GLuint gl_Tex;		// texture id
GLuint pbo = 0;     // pixel buffer object id

struct Configuration config;

// Size of image
int imgWidth = 0;
int imgHeight = 0;

// FPS variables and running time counters
int frame_count = 0;
GLfloat currenttime = 0;
GLfloat timebase = 0;
GLfloat starttime = 0;
//GLfloat runningtime = 60000.0f;
GLfloat runningtime = 15000.0f;

/*
	Initiate pixel buffer object
*/
void initPixelBuffer () {
    if (pbo) {
    	printf("pbo already set\n");
		// unmap pbo
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);

		// delete old buffers
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &gl_Tex);
    }

    // create a new pixel buffer object for display
	
	// size of pbo    
	int PBOsize = imgWidth * imgHeight * sizeof(float4);

	// create pixel buffer object
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, PBOsize, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// init texture object
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, imgWidth, imgHeight, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}

/*
	Rendering function. Run the solveFluid function, 
*/
static void Draw (void) {
	
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );

    // run fluid simulation
    solveFluid(&config);

    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imgWidth, imgHeight, GL_RGBA, GL_FLOAT, 0);	

	// OpenGL pointer to be passed to openacc kernel
	float4 *output; 
	output = (float4*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	
	// Host allocated pointer to make this work with pgi 13.3
	//float4 *out;
	//out = (float4*) malloc(imgWidth * imgHeight * sizeof(float4));

	if (output) {
		// get results from fluid simulation
		
		// Using OpenGL pointer
    	renderFluid(&config, output);

		// Using host allocated pointer
    	//renderFluid(&config, out);
    	//memcpy(output, out, imgWidth * imgHeight * sizeof(float4));
		
    	glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB);
	}
	else {printf("no output buffer\n");}

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	
    glutSwapBuffers();	    
}

/*
	Handle reshaping of window
*/
static void reshapeFunc (int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

/*
	Idle callback function. 
*/
static void idle_func (void) {
	// calculate FPS
	frame_count++;
	currenttime = glutGet(GLUT_ELAPSED_TIME);
	if (currenttime - timebase > 1000.0f) {
		printf("%4.2f\n", frame_count*1000/(currenttime-timebase));
		timebase = currenttime;
		frame_count = 0;
	}

	// end program if a set amount of time has passed.
	// this is for benchmarking purposes. 
	//if (currenttime - starttime > runningtime) {
	//	freeFluid( &config );
	//	exit(0);
	//}

	glutPostRedisplay();
}

/*
	Main function. 
	- Initiates glut, glew and the nvidia device used by OpenACC. 
	- Also initiates the pixel buffer object, sets the fluid dimensions
	- and initiates the allocation of memory used for the fluid simuation. 
*/
int main (int argc, char **argv) {
	printf("Starting simulation\n");

	// Image size
	int width = 512;
	int height = 512;
	
	// Init glut, the window, and glew
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( width, height );
	glutInitWindowPosition( 100, 100 );
    glutCreateWindow( "Fluid" );
	glewInit();	

	// Init OpenACC and the selected nvidia device
	acc_init(acc_device_nvidia);
	acc_set_device_num(0, acc_device_nvidia);

	// Set fluid dimension. does not have correspond to window size. 
	int fluidDim = 256;
	//int fluidDim = 512;
	//int fluidDim = 768;
	//int fluidDim = 1024;

	// we set the size of the rendered image to the same 
	// as the fluid dimension plus a boarder of 2. 
	imgWidth = fluidDim+2;
	imgHeight = fluidDim+2;	

	// Start timing
	starttime = glutGet(GLUT_ELAPSED_TIME);
	
	// Set OpenGL functions
	glutDisplayFunc( Draw );
	glutIdleFunc( idle_func );
	glutReshapeFunc( reshapeFunc );

	// Allocate memory for the fluid simulation
	initFluid( &config, fluidDim, fluidDim );

	// Initiate the pixel buffer object and texture
	initPixelBuffer();

	// OpenCL main loop
	glutMainLoop();

	// Free fluid simulation memory
	freeFluid( &config );
}
