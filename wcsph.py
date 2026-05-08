# Implementation based on https://github.com/InteractiveComputerGraphics/physics-simulation/blob/main/examples/wcsph.html

import math
import numpy as np
import taichi as ti
import time

ti.init(arch=ti.gpu, offline_cache=True, offline_cache_file_path="./taichi_cache")

WIDTH_PARTICLES = 20
HEIGHT_PARTICLES = 20

HASH_SIZE = 100_000
MAX_PARTICLES_PER_CELL = 128

CANVAS_W = 900
CANVAS_H = 700

FINAL_TIME = 8.0
DT = 0.002
TOTAL_SIMULATION_STEPS = int(FINAL_TIME / DT)

LOG_INTERVAL = 10


@ti.data_oriented
class WCSPHSimulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.particle_radius = 0.025
        self.support_radius = 4.0 * self.particle_radius
        self.density0 = 1000.0
        self.viscosity = 0.05
        self.diam = 2.0 * self.particle_radius
        self.mass_value = self.diam * self.diam * self.density0
        self.dt = DT
        self.stiffness = 35000.0
        self.exponent = 7.0
        self.gravity = -9.81

        self.bw = 3 * width
        self.bh = 3 * height

        self.num_fluid = width * height
        self.num_boundary = 2 * self.bw + 2 * (self.bh - 1)
        self.num_particles = self.num_fluid + self.num_boundary

        self.kernel_k = 40.0 / (
            7.0 * math.pi * self.support_radius * self.support_radius
        )
        self.kernel_l = 240.0 / (
            7.0 * math.pi * self.support_radius * self.support_radius
        )

        # Particle fields
        self.x = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.y = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.vx = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.vy = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.ax = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.ay = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.density = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.pressure = ti.field(dtype=ti.f32, shape=self.num_particles)

        # Density error diagnostics
        self.density_error_l1 = ti.field(dtype=ti.f32, shape=())
        self.density_error_linf = ti.field(dtype=ti.f32, shape=())
        self.density_error_percent = ti.field(dtype=ti.f32, shape=())

        # Kinetic energy diagnostics
        self.kinetic_energy = ti.field(dtype=ti.f32, shape=())

        # Boundary pseudo mass
        self.psi = ti.field(dtype=ti.f32, shape=self.num_particles)

        # Spatial hash grid
        self.grid_count = ti.field(dtype=ti.i32, shape=HASH_SIZE)
        self.grid_particles = ti.field(
            dtype=ti.i32, shape=(HASH_SIZE, MAX_PARTICLES_PER_CELL)
        )

        self.time = 0.0

        self.init_scene()
        self.precompute_boundary_psi()

    @ti.func
    def norm(self, x, y):
        return ti.sqrt(x * x + y * y)

    @ti.func
    def cubic_kernel_2d(self, r):
        res = 0.0
        q = r / self.support_radius

        if q <= 1.0:
            q2 = q * q
            q3 = q2 * q

            if q <= 0.5:
                res = self.kernel_k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                one_minus_q = 1.0 - q
                res = self.kernel_k * 2.0 * one_minus_q * one_minus_q * one_minus_q

        return res

    @ti.func
    def cubic_kernel_2d_gradient(self, rx, ry):
        gx = 0.0
        gy = 0.0

        rl = self.norm(rx, ry)
        q = rl / self.support_radius

        if q <= 1.0:
            if rl > 1.0e-6:
                gradq_x = rx / (rl * self.support_radius)
                gradq_y = ry / (rl * self.support_radius)

                if q <= 0.5:
                    factor = self.kernel_l * q * (3.0 * q - 2.0)
                    gx = factor * gradq_x
                    gy = factor * gradq_y
                else:
                    one_minus_q = 1.0 - q
                    factor = self.kernel_l * (-(one_minus_q * one_minus_q))
                    gx = factor * gradq_x
                    gy = factor * gradq_y

        return gx, gy

    @ti.func
    def hash_function(self, cx, cy):
        p1 = 73856093 * cx
        p2 = 19349663 * cy
        h = ti.abs(p1 + p2) % HASH_SIZE
        return h

    @ti.func
    def cell_x(self, x):
        return ti.cast(ti.floor((x + 100.0) / self.support_radius), ti.i32)

    @ti.func
    def cell_y(self, y):
        return ti.cast(ti.floor((y + 100.0) / self.support_radius), ti.i32)

    @ti.kernel
    def init_scene(self):
        # Fluid block
        for i, j in ti.ndrange(self.height, self.width):
            idx = i * self.width + j
            self.x[idx] = (
                -0.5 * self.bw * self.diam
                + j * self.diam
                + self.diam
                + self.particle_radius
            )
            self.y[idx] = i * self.diam + self.diam + self.particle_radius

            self.vx[idx] = 0.0
            self.vy[idx] = 0.0
            self.ax[idx] = 0.0
            self.ay[idx] = 0.0
            self.density[idx] = 0.0
            self.pressure[idx] = 0.0
            self.psi[idx] = 0.0

        # Boundary box
        offset = self.num_fluid

        # Bottom and top
        for j in range(self.bw):
            bottom_idx = offset + 2 * j
            top_idx = offset + 2 * j + 1

            px = -0.5 * self.bw * self.diam + j * self.diam

            self.x[bottom_idx] = px
            self.y[bottom_idx] = 0.0

            self.x[top_idx] = px
            self.y[top_idx] = self.bh * self.diam

            self.psi[bottom_idx] = 0.5
            self.psi[top_idx] = 0.5

        # Left and right
        side_offset = offset + 2 * self.bw
        for j in range(1, self.bh):
            k = j - 1

            left_idx = side_offset + 2 * k
            right_idx = side_offset + 2 * k + 1

            py = j * self.diam

            self.x[left_idx] = -0.5 * self.bw * self.diam
            self.y[left_idx] = py

            self.x[right_idx] = -0.5 * self.bw * self.diam + (self.bw - 1) * self.diam
            self.y[right_idx] = py

            self.psi[left_idx] = 0.5
            self.psi[right_idx] = 0.5

    @ti.kernel
    def precompute_boundary_psi(self):
        kernel_0 = self.cubic_kernel_2d(0.0)
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid, self.num_particles):
            delta = kernel_0

            for j in range(self.num_fluid, self.num_particles):
                if i != j:
                    dx = self.x[i] - self.x[j]
                    dy = self.y[i] - self.y[j]
                    dist2 = dx * dx + dy * dy

                    if dist2 <= radius2 + 1.0e-6:
                        delta += self.cubic_kernel_2d(ti.sqrt(dist2))

            self.psi[i] = self.density0 / delta

    @ti.kernel
    def clear_grid(self):
        for i in range(HASH_SIZE):
            self.grid_count[i] = 0

    @ti.kernel
    def build_grid(self):
        for i in range(self.num_particles):
            cx = self.cell_x(self.x[i])
            cy = self.cell_y(self.y[i])
            h = self.hash_function(cx, cy)

            slot = ti.atomic_add(self.grid_count[h], 1)

            if slot < MAX_PARTICLES_PER_CELL:
                self.grid_particles[h, slot] = i

    def update_grid(self):
        self.clear_grid()
        self.build_grid()

    @ti.kernel
    def reset_accelerations(self):
        for i in range(self.num_fluid):
            self.ax[i] = 0.0
            self.ay[i] = self.gravity

    @ti.kernel
    def compute_density(self):
        kernel_0 = self.cubic_kernel_2d(0.0)
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            rho = self.mass_value * kernel_0

            xi = self.x[i]
            yi = self.y[i]

            cx = self.cell_x(xi)
            cy = self.cell_y(yi)

            for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                h = self.hash_function(cx + ox, cy + oy)
                count = ti.min(self.grid_count[h], MAX_PARTICLES_PER_CELL)

                for slot in range(count):
                    j = self.grid_particles[h, slot]

                    if j != i:
                        dx = xi - self.x[j]
                        dy = yi - self.y[j]
                        dist2 = dx * dx + dy * dy

                        if dist2 <= radius2 + 1.0e-6:
                            Wij = self.cubic_kernel_2d(ti.sqrt(dist2))

                            if j < self.num_fluid:
                                rho += self.mass_value * Wij
                            else:
                                rho += self.psi[j] * Wij

            self.density[i] = rho

    @ti.kernel
    def compute_pressure(self):
        for i in range(self.num_fluid):
            rho = ti.max(self.density[i], self.density0)
            self.density[i] = rho

            self.pressure[i] = self.stiffness * (
                ti.pow(rho / self.density0, self.exponent) - 1.0
            )

    @ti.kernel
    def compute_pressure_accelerations(self):
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]

            dpi = self.pressure[i] / (self.density[i] * self.density[i])

            ax_i = self.ax[i]
            ay_i = self.ay[i]

            cx = self.cell_x(xi)
            cy = self.cell_y(yi)

            for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                h = self.hash_function(cx + ox, cy + oy)
                count = ti.min(self.grid_count[h], MAX_PARTICLES_PER_CELL)

                for slot in range(count):
                    j = self.grid_particles[h, slot]

                    if j != i:
                        rx = xi - self.x[j]
                        ry = yi - self.y[j]
                        dist2 = rx * rx + ry * ry

                        if dist2 <= radius2 + 1.0e-6:
                            grad_x, grad_y = self.cubic_kernel_2d_gradient(rx, ry)

                            if j < self.num_fluid:
                                dpj = self.pressure[j] / (
                                    self.density[j] * self.density[j]
                                )

                                factor = self.mass_value * (dpi + dpj)

                                ax_i -= factor * grad_x
                                ay_i -= factor * grad_y
                            else:
                                factor = self.psi[j] * dpi

                                ax_i -= factor * grad_x
                                ay_i -= factor * grad_y

            self.ax[i] = ax_i
            self.ay[i] = ay_i

    @ti.kernel
    def compute_viscosity(self):
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]

            vxi = self.vx[i]
            vyi = self.vy[i]

            ax_i = self.ax[i]
            ay_i = self.ay[i]

            cx = self.cell_x(xi)
            cy = self.cell_y(yi)

            for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                h = self.hash_function(cx + ox, cy + oy)
                count = ti.min(self.grid_count[h], MAX_PARTICLES_PER_CELL)

                for slot in range(count):
                    j = self.grid_particles[h, slot]

                    if j != i and j < self.num_fluid:
                        dx = xi - self.x[j]
                        dy = yi - self.y[j]
                        dist2 = dx * dx + dy * dy

                        if dist2 <= radius2 + 1.0e-6:
                            Wij = self.cubic_kernel_2d(ti.sqrt(dist2))

                            vij_x = vxi - self.vx[j]
                            vij_y = vyi - self.vy[j]

                            factor = (
                                self.mass_value
                                / self.density[j]
                                * (1.0 / self.dt)
                                * self.viscosity
                                * Wij
                            )

                            ax_i -= factor * vij_x
                            ay_i -= factor * vij_y

            self.ax[i] = ax_i
            self.ay[i] = ay_i

    @ti.kernel
    def symplectic_euler(self):
        for i in range(self.num_fluid):
            self.vx[i] += self.dt * self.ax[i]
            self.vy[i] += self.dt * self.ay[i]

            self.x[i] += self.dt * self.vx[i]
            self.y[i] += self.dt * self.vy[i]

    @ti.kernel
    def compute_density_error(self):
        self.density_error_l1[None] = 0.0
        self.density_error_linf[None] = 0.0

        for i in range(self.num_fluid):
            err = ti.abs(self.density[i] - self.density0) / self.density0

            ti.atomic_add(self.density_error_l1[None], err)
            ti.atomic_max(self.density_error_linf[None], err)

        self.density_error_l1[None] /= ti.cast(self.num_fluid, ti.f32)
        self.density_error_percent[None] = self.density_error_l1[None] * 100.0

    def get_density_error(self):
        self.update_grid()
        self.compute_density()
        self.compute_density_error()
        ti.sync()

        return {
            "l1": float(self.density_error_l1[None]),
            "linf": float(self.density_error_linf[None]),
            "percent": float(self.density_error_percent[None]),
        }

    @ti.kernel
    def compute_kinetic_energy(self):
        self.kinetic_energy[None] = 0.0

        for i in range(self.num_fluid):
            v2 = self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i]
            ti.atomic_add(
                self.kinetic_energy[None],
                0.5 * self.mass_value * v2
            )
    
    def get_kinetic_energy(self):
        self.compute_kinetic_energy()
        ti.sync()
        return float(self.kinetic_energy[None])

    def simulation_step(self):
        self.reset_accelerations()
        self.update_grid()
        self.compute_density()
        self.compute_pressure()
        self.compute_pressure_accelerations()
        self.compute_viscosity()
        self.symplectic_euler()
        self.time += self.dt

    def particle_positions_numpy(self):
        x = self.x.to_numpy()
        y = self.y.to_numpy()

        screen = np.zeros((self.num_particles, 2), dtype=np.float32)

        origin_x = CANVAS_W / 2
        origin_y = CANVAS_H / 2 + 200
        zoom = 100.0

        screen[:, 0] = (origin_x + x * zoom) / CANVAS_W
        screen[:, 1] = 1.0 - (origin_y - y * zoom) / CANVAS_H

        return screen


def main():
    sim = WCSPHSimulation(WIDTH_PARTICLES, HEIGHT_PARTICLES)

    gui = ti.GUI(
        "WCSPH Fluid",
        res=(CANVAS_W, CANVAS_H),
        background_color=0xFFFFFF
    )

    paused = False
    steps_per_frame = 8

    # Warm start
    for _ in range(3):
        sim.simulation_step()
    ti.sync()

    # # Logging density error to CSV
    # log_file = open("density_error_wcsph_dt2ms.csv", "w")
    # log_file.write("step,time,density_error_avg_percent,density_error_max_percent\n")

    # density_err = {
    #     "percent": 0.0,
    #     "linf": 0.0,
    # }

    # # Logging kinetic energy to CSV
    # log_file = open("kinetic_energy_wcsph.csv", "w")
    # log_file.write("step,time,kinetic_energy\n")

    #kinetic_energy = 0.0

    current_step = 0
    compute_ms = 0.0
    step_ms = 0.0
    total_compute_ms = 0.0
    average_step_ms = 0.0

    while gui.running:
        for event in gui.get_events():
            if event.key == gui.ESCAPE:
                gui.running = False

            if event.key == gui.SPACE and event.type == ti.GUI.PRESS:
                paused = not paused

            if event.key == "r" and event.type == ti.GUI.PRESS:
                sim = WCSPHSimulation(WIDTH_PARTICLES, HEIGHT_PARTICLES)
                
                for _ in range(3):
                    sim.simulation_step()
                ti.sync()

                # kinetic_energy = 0.0
                current_step = 0
                compute_ms = 0.0
                step_ms = 0.0
                total_compute_ms = 0.0
                average_step_ms = 0.0

        if not paused and current_step < TOTAL_SIMULATION_STEPS:
            start = time.perf_counter()

            for _ in range(steps_per_frame):
                sim.simulation_step()
                current_step += 1

                # if current_step > 0 and current_step % LOG_INTERVAL == 0:
                #     kinetic_energy = sim.get_kinetic_energy()
                #     log_file.write(
                #         f"{current_step},"
                #         f"{sim.time},"
                #         f"{kinetic_energy}\n"
                #     )
                #     log_file.flush()
            
            ti.sync()
            end = time.perf_counter()

            compute_ms = (end - start) * 1000.0
            step_ms = compute_ms / steps_per_frame

            total_compute_ms += compute_ms
            average_step_ms = total_compute_ms / current_step

            

        pos = sim.particle_positions_numpy()
        
        # # Log density error every LOG_INTERVAL steps
        # if current_step > 0 and current_step % LOG_INTERVAL == 0:
        #     density_err = sim.get_density_error()

        #     log_file.write(
        #         f"{current_step},"
        #         f"{sim.time},"
        #         f"{density_err['percent']},"
        #         f"{density_err['linf']*100.0}\n"
        #     )
        #     log_file.flush()




        fluid_pos = pos[: sim.num_fluid]
        boundary_pos = pos[sim.num_fluid : sim.num_particles]

        gui.circles(boundary_pos, radius=3, color=0x888888)
        gui.circles(fluid_pos, radius=3, color=0x1E6CFF)

        gui.text(
            f"time = {sim.time:.3f} | particles = {sim.num_particles} | compute = {compute_ms:.2f} ms | step = {step_ms:.2f} ms |SPACE pause | R reset",
            pos=(0.02, 0.96),
            color=0x000000,
        )   
        gui.text(
            f"avg step ms = {average_step_ms:.2f} ms",
            pos=(0.02, 0.92),
            color=0x000000,
        ) 

        gui.show()

if __name__ == "__main__":
    main()