import moderngl_window as mglw
import moderngl
import numpy as np
import glm

class CubeWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Cube Render Test"
    window_size = (800, 600)
    aspect_ratio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Vertex shader: transforms 3D points using model/view/projection matrices
        vertex_shader = '''
        #version 330
        uniform mat4 mvp;
        in vec3 in_position;
        in vec3 in_color;
        out vec3 color;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            color = in_color;
        }
        '''

        # Fragment shader: just outputs the interpolated color
        fragment_shader = '''
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        '''

        self.prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Cube vertices with distinct colors per face (helps visually confirm real 3D)
        vertices = np.array([
            # positions          # colors
            -0.5,-0.5,-0.5,  1,0,0,
             0.5,-0.5,-0.5,  1,0,0,
             0.5, 0.5,-0.5,  1,0,0,
            -0.5,-0.5,-0.5,  1,0,0,
             0.5, 0.5,-0.5,  1,0,0,
            -0.5, 0.5,-0.5,  1,0,0,

            -0.5,-0.5, 0.5,  0,1,0,
             0.5,-0.5, 0.5,  0,1,0,
             0.5, 0.5, 0.5,  0,1,0,
            -0.5,-0.5, 0.5,  0,1,0,
             0.5, 0.5, 0.5,  0,1,0,
            -0.5, 0.5, 0.5,  0,1,0,

            -0.5,-0.5,-0.5,  0,0,1,
            -0.5, 0.5,-0.5,  0,0,1,
            -0.5, 0.5, 0.5,  0,0,1,
            -0.5,-0.5,-0.5,  0,0,1,
            -0.5, 0.5, 0.5,  0,0,1,
            -0.5,-0.5, 0.5,  0,0,1,

             0.5,-0.5,-0.5,  1,1,0,
             0.5, 0.5,-0.5,  1,1,0,
             0.5, 0.5, 0.5,  1,1,0,
             0.5,-0.5,-0.5,  1,1,0,
             0.5, 0.5, 0.5,  1,1,0,
             0.5,-0.5, 0.5,  1,1,0,

            -0.5,-0.5,-0.5,  1,0,1,
             0.5,-0.5,-0.5,  1,0,1,
             0.5,-0.5, 0.5,  1,0,1,
            -0.5,-0.5,-0.5,  1,0,1,
             0.5,-0.5, 0.5,  1,0,1,
            -0.5,-0.5, 0.5,  1,0,1,

            -0.5, 0.5,-0.5,  0,1,1,
             0.5, 0.5,-0.5,  0,1,1,
             0.5, 0.5, 0.5,  0,1,1,
            -0.5, 0.5,-0.5,  0,1,1,
             0.5, 0.5, 0.5,  0,1,1,
            -0.5, 0.5, 0.5,  0,1,1,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '3f 3f', 'in_position', 'in_color')]
        )

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.angle = 0.0

    def on_render(self, time, frame_time):
        self.ctx.clear(0.1, 0.1, 0.1)
        self.angle += frame_time  # rotate over time so we can SEE it's really 3D

        model = glm.rotate(glm.mat4(1.0), self.angle, glm.vec3(0.3, 1.0, 0.2))
        view = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, -3.0))
        proj = glm.perspective(glm.radians(45.0), self.wnd.aspect_ratio, 0.1, 100.0)

        mvp = proj * view * model
        self.prog['mvp'].write(mvp.to_bytes())

        self.vao.render(moderngl.TRIANGLES)

if __name__ == '__main__':
    mglw.run_window_config(CubeWindow)
