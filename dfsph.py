# Implementation based on https://github.com/InteractiveComputerGraphics/physics-simulation/blob/main/examples/dfsph.html

import math
import numpy as np
import taichi as ti
import time

ti.init(arch=ti.gpu)

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
class DFSPHSimulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.particle_radius = 0.025
        self.support_radius = 4.0 * self.particle_radius
        self.density0 = 1000.0
        self.viscosity = 0.01
        self.diam = 2.0 * self.particle_radius
        self.mass_value = self.diam * self.diam * self.density0
        self.dt = DT

        self.max_iterations = 100
        self.max_iterations_v = 100
        self.max_error = 0.01
        self.max_error_v = 0.1
        self.gravity = -9.81

        self.time = 0.0
        self.iterations_used = 0
        self.iterations_v_used = 0
        self.avg_density_error = 0.0
        self.avg_divergence_error = 0.0

        self.bw = 3 * width
        self.bh = 3 * height

        self.num_fluid = width * height
        self.num_boundary = 2 * self.bw + 2 * (self.bh - 1)
        self.num_particles = self.num_fluid + self.num_boundary

        self.kernel_k = 40.0 / (7.0 * math.pi * self.support_radius * self.support_radius)
        self.kernel_l = 240.0 / (7.0 * math.pi * self.support_radius * self.support_radius)

        # Particle fields
        self.x = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.y = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.vx = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.vy = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.ax = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.ay = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.pax = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.pay = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.density = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.source_term = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.pressure = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.pressure_v = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.aii = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.aij_pj = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.psi = ti.field(dtype=ti.f32, shape=self.num_particles)

        self.neighbor_count = ti.field(dtype=ti.i32, shape=self.num_particles)

        # Density error diagnostics
        self.density_error_l1 = ti.field(dtype=ti.f32, shape=())
        self.density_error_linf = ti.field(dtype=ti.f32, shape=())
        self.density_error_percent = ti.field(dtype=ti.f32, shape=())

        # Kinetic energy diagnostics
        self.kinetic_energy = ti.field(dtype=ti.f32, shape=())

        # Pressure diagnostics
        self.pressure_avg = ti.field(dtype=ti.f32, shape=())
        self.pressure_max = ti.field(dtype=ti.f32, shape=())
        self.pressure_min = ti.field(dtype=ti.f32, shape=())

        self.grid_count = ti.field(dtype=ti.i32, shape=HASH_SIZE)
        self.grid_particles = ti.field(dtype=ti.i32, shape=(HASH_SIZE, MAX_PARTICLES_PER_CELL))

        self.init_scene()
        self.precompute_boundary_psi()

        self.update_grid()
        self.compute_neighbor_count()
        self.compute_density()
        self.compute_aii()

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
        return ti.abs(p1 + p2) % HASH_SIZE

    @ti.func
    def cell_x(self, x):
        return ti.cast(ti.floor((x + 100.0) / self.support_radius), ti.i32)

    @ti.func
    def cell_y(self, y):
        return ti.cast(ti.floor((y + 100.0) / self.support_radius), ti.i32)

    @ti.kernel
    def init_scene(self):
        for i, j in ti.ndrange(self.height, self.width):
            idx = i * self.width + j

            self.x[idx] = -0.5 * self.bw * self.diam + j * self.diam + self.diam + self.particle_radius
            self.y[idx] = i * self.diam + self.diam + self.particle_radius

            self.vx[idx] = 0.0
            self.vy[idx] = 0.0
            self.ax[idx] = 0.0
            self.ay[idx] = 0.0
            self.pax[idx] = 0.0
            self.pay[idx] = 0.0
            self.density[idx] = 0.0
            self.source_term[idx] = 0.0
            self.pressure[idx] = 0.0
            self.pressure_v[idx] = 0.0
            self.aii[idx] = 0.0
            self.aij_pj[idx] = 0.0
            self.psi[idx] = 0.0
            self.neighbor_count[idx] = 0

        offset = self.num_fluid

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
    def compute_neighbor_count(self):
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            cx = self.cell_x(xi)
            cy = self.cell_y(yi)
            count_neighbors = 0

            for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                h = self.hash_function(cx + ox, cy + oy)
                count = ti.min(self.grid_count[h], MAX_PARTICLES_PER_CELL)

                for slot in range(count):
                    j = self.grid_particles[h, slot]
                    if j != i:
                        dx = xi - self.x[j]
                        dy = yi - self.y[j]
                        if dx * dx + dy * dy <= radius2 + 1.0e-6:
                            count_neighbors += 1

            self.neighbor_count[i] = count_neighbors

    @ti.kernel
    def reset_accelerations(self):
        for i in range(self.num_fluid):
            self.ax[i] = 0.0
            self.ay[i] = self.gravity
            self.pax[i] = 0.0
            self.pay[i] = 0.0

    @ti.kernel
    def compute_density(self):
        kernel_0 = self.cubic_kernel_2d(0.0)
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            rho = self.mass_value * kernel_0

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
    def compute_aii(self):
        dt = self.dt
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]

            sum_m_gradW = 0.0
            m_gradW_x = 0.0
            m_gradW_y = 0.0

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
                                mgw_x = self.mass_value * grad_x
                                mgw_y = self.mass_value * grad_y
                                sum_m_gradW += mgw_x * mgw_x + mgw_y * mgw_y
                                m_gradW_x += mgw_x
                                m_gradW_y += mgw_y
                            else:
                                mgw_x = self.psi[j] * grad_x
                                mgw_y = self.psi[j] * grad_y
                                m_gradW_x += mgw_x
                                m_gradW_y += mgw_y

            sum_m_gradW += m_gradW_x * m_gradW_x + m_gradW_y * m_gradW_y
            self.aii[i] = -dt / (self.density[i] * self.density[i]) * sum_m_gradW

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
                            vi_vj_x = vxi - self.vx[j]
                            vi_vj_y = vyi - self.vy[j]
                            factor = self.mass_value / self.density[j] * (1.0 / self.dt) * self.viscosity * Wij
                            ax_i -= factor * vi_vj_x
                            ay_i -= factor * vi_vj_y

            self.ax[i] = ax_i
            self.ay[i] = ay_i

    @ti.kernel
    def integrate_non_pressure_velocity(self):
        for i in range(self.num_fluid):
            self.vx[i] += self.dt * self.ax[i]
            self.vy[i] += self.dt * self.ay[i]

    @ti.kernel
    def compute_constant_density_source_term(self):
        dt = self.dt
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            vxi = self.vx[i]
            vyi = self.vy[i]
            Drho_Dt = 0.0

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
                                Drho_Dt += self.mass_value * ((vxi - self.vx[j]) * grad_x + (vyi - self.vy[j]) * grad_y)
                            else:
                                Drho_Dt += self.psi[j] * (vxi * grad_x + vyi * grad_y)

            self.source_term[i] = (1.0 / dt) * (self.density0 - (self.density[i] + dt * Drho_Dt))

    @ti.kernel
    def compute_divergence_source_term(self):
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            vxi = self.vx[i]
            vyi = self.vy[i]
            Drho_Dt = 0.0

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
                                Drho_Dt += self.mass_value * ((vxi - self.vx[j]) * grad_x + (vyi - self.vy[j]) * grad_y)
                            else:
                                Drho_Dt += self.psi[j] * (vxi * grad_x + vyi * grad_y)

            self.source_term[i] = -Drho_Dt

    @ti.kernel
    def compute_pressure_accelerations_density(self):
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            dpi = self.pressure[i] / (self.density[i] * self.density[i])
            pax_i = 0.0
            pay_i = 0.0

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
                                dpj = self.pressure[j] / (self.density[j] * self.density[j])
                                factor = self.mass_value * (dpi + dpj)
                                pax_i -= factor * grad_x
                                pay_i -= factor * grad_y
                            else:
                                factor = self.psi[j] * dpi
                                pax_i -= factor * grad_x
                                pay_i -= factor * grad_y

            self.pax[i] = pax_i
            self.pay[i] = pay_i

    @ti.kernel
    def compute_pressure_accelerations_divergence(self):
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            dpi = self.pressure_v[i] / (self.density[i] * self.density[i])
            pax_i = 0.0
            pay_i = 0.0

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
                                dpj = self.pressure_v[j] / (self.density[j] * self.density[j])
                                factor = self.mass_value * (dpi + dpj)
                                pax_i -= factor * grad_x
                                pay_i -= factor * grad_y
                            else:
                                factor = self.psi[j] * dpi
                                pax_i -= factor * grad_x
                                pay_i -= factor * grad_y

            self.pax[i] = pax_i
            self.pay[i] = pay_i

    @ti.kernel
    def compute_aij_pj(self):
        dt = self.dt
        radius2 = self.support_radius * self.support_radius

        for i in range(self.num_fluid):
            xi = self.x[i]
            yi = self.y[i]
            val = 0.0

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
                                val += self.mass_value * ((self.pax[i] - self.pax[j]) * grad_x + (self.pay[i] - self.pay[j]) * grad_y)
                            else:
                                val += self.psi[j] * (self.pax[i] * grad_x + self.pay[i] * grad_y)

            self.aij_pj[i] = val * dt

    @ti.kernel
    def update_pressure_density(self) -> ti.f32:
        density_err = 0.0

        for i in range(self.num_fluid):
            residual = self.source_term[i] - self.aij_pj[i]

            if ti.abs(self.aii[i]) > 1.0e-6:
                self.pressure[i] += 0.5 / self.aii[i] * residual
            else:
                self.pressure[i] = 0.0

            self.pressure[i] = ti.max(self.pressure[i], 0.0)
            density_err -= ti.min(residual, 0.0) * self.dt

        return density_err / self.num_fluid

    @ti.kernel
    def update_pressure_divergence(self) -> ti.f32:
        density_err = 0.0

        for i in range(self.num_fluid):
            residual = self.source_term[i] - self.aij_pj[i]

            if ti.abs(self.aii[i]) > 1.0e-6:
                self.pressure_v[i] += 0.5 / self.aii[i] * residual
            else:
                self.pressure_v[i] = 0.0

            if self.neighbor_count[i] < 7:
                self.pressure_v[i] = 0.0

            self.pressure_v[i] = ti.max(self.pressure_v[i], 0.0)
            density_err -= ti.min(residual, 0.0) * self.dt

        return density_err / self.num_fluid

    @ti.kernel
    def integrate_density_pressure_and_position(self):
        for i in range(self.num_fluid):
            self.vx[i] += self.dt * self.pax[i]
            self.vy[i] += self.dt * self.pay[i]
            self.x[i] += self.dt * self.vx[i]
            self.y[i] += self.dt * self.vy[i]

    @ti.kernel
    def integrate_divergence_pressure_velocity(self):
        for i in range(self.num_fluid):
            self.vx[i] += self.dt * self.pax[i]
            self.vy[i] += self.dt * self.pay[i]

    def constant_density_solve(self):
        self.avg_density_error = 1000.0
        self.iterations_used = 0
        threshold = self.max_error * 0.01 * self.density0

        while ((self.avg_density_error > threshold and self.iterations_used < self.max_iterations) or self.iterations_used < 2):
            self.compute_pressure_accelerations_density()
            self.compute_aij_pj()
            self.avg_density_error = float(self.update_pressure_density())
            self.iterations_used += 1

    def divergence_solve(self):
        self.avg_divergence_error = 1000.0
        self.iterations_v_used = 0
        threshold = self.max_error_v * 0.01 * self.density0

        while ((self.avg_divergence_error > threshold and self.iterations_v_used < self.max_iterations_v) or self.iterations_v_used < 1):
            self.compute_pressure_accelerations_divergence()
            self.compute_aij_pj()
            self.avg_divergence_error = float(self.update_pressure_divergence())
            self.iterations_v_used += 1

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


    @ti.kernel
    def compute_pressure_stats(self):
        self.pressure_avg[None] = 0.0
        self.pressure_max[None] = 0.0
        self.pressure_min[None] = 1.0e20

        for i in range(self.num_fluid):
            p = self.pressure[i]

            ti.atomic_add(self.pressure_avg[None], p)
            ti.atomic_max(self.pressure_max[None], p)
            ti.atomic_min(self.pressure_min[None], p)

        self.pressure_avg[None] /= ti.cast(self.num_fluid, ti.f32)


    def get_pressure_stats(self):
        self.compute_pressure_stats()
        ti.sync()

        return {
            "avg": float(self.pressure_avg[None]),
            "max": float(self.pressure_max[None]),
            "min": float(self.pressure_min[None]),
        }

    def simulation_step(self):
        self.reset_accelerations()

        self.compute_viscosity()
        self.integrate_non_pressure_velocity()

        self.compute_constant_density_source_term()
        self.constant_density_solve()

        self.integrate_density_pressure_and_position()
        self.time += self.dt

        self.update_grid()
        self.compute_neighbor_count()
        self.compute_density()
        self.compute_aii()

        self.compute_divergence_source_term()
        self.divergence_solve()

        self.integrate_divergence_pressure_velocity()

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
    sim = DFSPHSimulation(WIDTH_PARTICLES, HEIGHT_PARTICLES)

    gui = ti.GUI(
        "DFSPH Fluid",
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
    # log_file = open("density_error_dfsph_dt2ms.csv", "w")
    # log_file.write("step,time,density_error_avg_percent,density_error_max_percent\n")

    # density_err = {
    #     "percent": 0.0,
    #     "linf": 0.0,
    # }

    # # Logging kinetic energy to CSV
    # log_file = open("kinetic_energy_dfsph.csv", "w")
    # log_file.write("step,time,kinetic_energy\n")

    # kinetic_energy = 0.0

    # # Logging pressure stats to CSV
    # pressure_log_file = open("pressure_stats_dfsph.csv", "w")
    # pressure_log_file.write("step,time,pressure_avg,pressure_max,pressure_min\n")


    current_step = 0
    compute_ms = 0.0
    step_ms = 0.0
    total_compute_ms = 0.0
    average_step_ms = 0.0

    while gui.running:
        for event in gui.get_events():
            if event.key == gui.ESCAPE:
                gui.running = False

            elif event.key == gui.SPACE and event.type == ti.GUI.PRESS:
                paused = not paused

            elif event.key == "r" and event.type == ti.GUI.PRESS:
                sim = DFSPHSimulation(WIDTH_PARTICLES, HEIGHT_PARTICLES)
            
                for _ in range(3):
                    sim.simulation_step()
                ti.sync()

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

                # if current_step > 0 and current_step % LOG_INTERVAL == 0:
                #     pressure_stats = sim.get_pressure_stats()

                #     pressure_log_file.write(
                #         f"{current_step},"
                #         f"{sim.time},"
                #         f"{pressure_stats['avg']},"
                #         f"{pressure_stats['max']},"
                #         f"{pressure_stats['min']}\n"
                #     )
                #     pressure_log_file.flush()        
                
            ti.sync()
            end = time.perf_counter()
            compute_ms = (end - start) * 1000.0
            step_ms = compute_ms / steps_per_frame

            total_compute_ms += compute_ms
            average_step_ms = total_compute_ms / current_step

        pos = sim.particle_positions_numpy()
        fluid_pos = pos[: sim.num_fluid]
        boundary_pos = pos[sim.num_fluid : sim.num_particles]

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

        gui.circles(boundary_pos, radius=3, color=0x888888)
        gui.circles(fluid_pos, radius=3, color=0x1E6CFF)

        gui.text(
            f"time = {sim.time:.3f} | compute = {compute_ms:.2f} ms | step = {step_ms:.2f} ms",
            pos=(0.02, 0.96),
            color=0x000000,
        )
        gui.text(
            f"particles = {sim.num_particles} | fluid = {sim.num_fluid}",
            pos=(0.02, 0.92),
            color=0x000000
        )
        gui.text(
            f"density iterations = {sim.iterations_used} | err = {sim.avg_density_error:.5f}",
            pos=(0.02, 0.88),
            color=0x000000
        )
        gui.text(
            f"divergence iterations = {sim.iterations_v_used} | err = {sim.avg_divergence_error:.5f}",
            pos=(0.02, 0.84),
            color=0x000000
        )
        gui.text(
            "SPACE pause | R reset | ESC quit",
            pos=(0.02, 0.80),
            color=0x000000
        )
        gui.text(
            f"avg step ms = {average_step_ms:.2f} ms",
            pos=(0.02, 0.76),
            color=0x000000
        )

        gui.show()

if __name__ == "__main__":
    main()
