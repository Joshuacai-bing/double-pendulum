import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import csv
import time

class DoublePendulumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("理想双摆物理模拟系统 (RK4)")
        self.root.geometry("1400x800")
        self.root.configure(bg='#ecf0f1')
        
        # State
        self.is_running = False
        self.t = 0.0
        self.state = [math.pi/2, 0.0, math.pi/2, 0.0]
        self.params = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81, 'dt': 0.01, 'method': 'rk4'}
        self.initial_energy = 0.0
        
        self.trail_data = []
        self.plot_data = []
        self.last_record_time = 0.0
        self.MAX_TRAIL = 150
        self.MAX_PLOT_POINTS = 500
        
        self.dragging_node = 0
        self.last_frame_time = time.perf_counter()
        self.time_accumulator = 0.0
        
        self.setup_ui()
        self.reset_simulation()
        self.update_loop()

    def setup_ui(self):
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        # Left Panel (Controls)
        left_panel = tk.Frame(self.root, width=350, bg='white', padx=20, pady=20)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)
        
        tk.Label(left_panel, text="⚙️ 双摆系统参数配置", font=("Segoe UI", 16, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0,10))
        
        self.vars = {
            'm1': tk.DoubleVar(value=1.0),
            'm2': tk.DoubleVar(value=1.0),
            'l1': tk.DoubleVar(value=1.0),
            'l2': tk.DoubleVar(value=1.0),
            'g': tk.DoubleVar(value=9.8),
            'th1': tk.DoubleVar(value=90.0),
            'th2': tk.DoubleVar(value=90.0),
            'dt': tk.DoubleVar(value=0.01)
        }
        
        for key in self.vars:
            self.vars[key].trace_add('write', self.on_var_change)
            
        def create_slider_group(parent, label_text, var, from_, to, res):
            frame = tk.Frame(parent, bg='white')
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=label_text, bg='white', font=("Segoe UI", 10, "bold"), fg='#333').pack(anchor='w')
            
            row = tk.Frame(frame, bg='white')
            row.pack(fill=tk.X)
            scale = tk.Scale(row, from_=from_, to=to, resolution=res, variable=var, orient=tk.HORIZONTAL, bg='white', highlightthickness=0, showvalue=0)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            entry = ttk.Entry(row, textvariable=var, width=8)
            entry.pack(side=tk.RIGHT)
            
        tk.Label(left_panel, text="物理参数", font=("Segoe UI", 12, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(10,5))
        create_slider_group(left_panel, "摆1 质量 m₁ (kg)", self.vars['m1'], 0.1, 10.0, 0.1)
        create_slider_group(left_panel, "摆2 质量 m₂ (kg)", self.vars['m2'], 0.1, 10.0, 0.1)
        create_slider_group(left_panel, "摆1 绳长 L₁ (m)", self.vars['l1'], 0.5, 5.0, 0.1)
        create_slider_group(left_panel, "摆2 绳长 L₂ (m)", self.vars['l2'], 0.5, 5.0, 0.1)
        create_slider_group(left_panel, "重力加速度 g (m/s²)", self.vars['g'], 1.0, 25.0, 0.1)
        
        tk.Label(left_panel, text="初始条件 (可直接在画布拖拽)", font=("Segoe UI", 12, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(10,5))
        create_slider_group(left_panel, "摆1 初始角度 θ₁ (°)", self.vars['th1'], -180, 180, 1)
        create_slider_group(left_panel, "摆2 初始角度 θ₂ (°)", self.vars['th2'], -180, 180, 1)
        
        tk.Label(left_panel, text="模拟设置", font=("Segoe UI", 12, "bold"), bg='white', fg='#2c3e50').pack(anchor='w', pady=(10,5))
        
        m_frame = tk.Frame(left_panel, bg='white')
        m_frame.pack(fill=tk.X, pady=2)
        tk.Label(m_frame, text="积分方法", bg='white', font=("Segoe UI", 10, "bold")).pack(anchor='w')
        self.method_var = tk.StringVar(value='rk4')
        combo = ttk.Combobox(m_frame, textvariable=self.method_var, values=['rk4', 'euler'], state="readonly")
        combo.pack(fill=tk.X)
        self.method_var.trace_add('write', self.on_var_change)
        
        dt_frame = tk.Frame(left_panel, bg='white')
        dt_frame.pack(fill=tk.X, pady=(5, 0))
        tk.Label(dt_frame, text="时间步长 dt (s)", bg='white', font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Entry(dt_frame, textvariable=self.vars['dt'], width=8).pack(side=tk.RIGHT)
        tk.Label(left_panel, text="较小的dt能提高精度，但也需要更多计算量", bg='white', fg='#7f8c8d', font=("Segoe UI", 8)).pack(anchor='w')
        
        btn_frame = tk.Frame(left_panel, bg='white')
        btn_frame.pack(fill=tk.X, pady=(20, 5))
        
        self.btn_play = tk.Button(btn_frame, text="开始 / 暂停", bg='#2ecc71', fg='white', font=("Segoe UI", 11, "bold"), relief=tk.FLAT, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=5)
        
        self.btn_reset = tk.Button(btn_frame, text="重置模拟", bg='#e74c3c', fg='white', font=("Segoe UI", 11, "bold"), relief=tk.FLAT, command=self.btn_reset_clicked)
        self.btn_reset.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0), ipady=5)
        
        self.btn_export = tk.Button(left_panel, text="导出 CSV 数据", bg='#3498db', fg='white', font=("Segoe UI", 11, "bold"), relief=tk.FLAT, command=self.btn_export_clicked)
        self.btn_export.pack(fill=tk.X, pady=5, ipady=5)
        
        self.stats_label = tk.Label(left_panel, text="", bg='#1e1e1e', fg='#00ff00', font=("Courier New", 10), justify=tk.LEFT, anchor='nw', padx=15, pady=15)
        self.stats_label.pack(fill=tk.X, pady=(15, 0))
        
        # Right Panel (Visuals)
        right_panel = tk.Frame(self.root, bg='#ecf0f1', padx=20, pady=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Pendulum Canvas
        pen_frame = tk.Frame(right_panel, bg='white', bd=0)
        pen_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        self.pen_canvas = tk.Canvas(pen_frame, bg='#fafafa', highlightthickness=1, highlightbackground='#ddd')
        self.pen_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(pen_frame, text="💡 提示：在暂停或重置状态下，可以直接用鼠标拖拽小球来设置初始角度。", bg='white', fg='#7f8c8d', font=("Segoe UI", 10)).pack(pady=(0,10))
        
        self.pen_canvas.bind("<Button-1>", self.on_mouse_down)
        self.pen_canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.pen_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.pen_canvas.bind("<Configure>", lambda e: self.draw_pendulum())
        
        # Plot Canvas
        plot_frame = tk.Frame(right_panel, bg='white', bd=0)
        plot_frame.pack(fill=tk.X)
        
        legend_frame = tk.Frame(plot_frame, bg='white')
        legend_frame.pack(pady=(10,0))
        
        l1 = tk.Frame(legend_frame, bg='white')
        l1.pack(side=tk.LEFT, padx=15)
        tk.Label(l1, width=2, bg='#e74c3c').pack(side=tk.LEFT, padx=(0,5))
        tk.Label(l1, text="θ₁ (角度)", bg='white', font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        l2 = tk.Frame(legend_frame, bg='white')
        l2.pack(side=tk.LEFT, padx=15)
        tk.Label(l2, width=2, bg='#3498db').pack(side=tk.LEFT, padx=(0,5))
        tk.Label(l2, text="θ₂ (角度)", bg='white', font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        self.plot_canvas = tk.Canvas(plot_frame, height=250, bg='white', highlightthickness=1, highlightbackground='#ddd')
        self.plot_canvas.pack(fill=tk.X, padx=10, pady=(10, 10))
        self.plot_canvas.bind("<Configure>", lambda e: self.draw_plot())

    def get_derivatives(self, s):
        th1, w1, th2, w2 = s
        m1, m2, l1, l2, g = self.params['m1'], self.params['m2'], self.params['l1'], self.params['l2'], self.params['g']
        
        delta = th1 - th2
        den = 2 * m1 + m2 - m2 * math.cos(2 * delta)
        
        num1 = -g * (2 * m1 + m2) * math.sin(th1) \
               - m2 * g * math.sin(th1 - 2 * th2) \
               - 2 * math.sin(delta) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(delta))
        alpha1 = num1 / (l1 * den)
        
        num2 = 2 * math.sin(delta) * (
                w1 * w1 * l1 * (m1 + m2) 
                + g * (m1 + m2) * math.cos(th1) 
                + w2 * w2 * l2 * m2 * math.cos(delta)
               )
        alpha2 = num2 / (l2 * den)
        
        return [w1, alpha1, w2, alpha2]

    def rk4_step(self, s, dt):
        k1 = self.get_derivatives(s)
        s2 = [s[i] + 0.5 * dt * k1[i] for i in range(4)]
        k2 = self.get_derivatives(s2)
        s3 = [s[i] + 0.5 * dt * k2[i] for i in range(4)]
        k3 = self.get_derivatives(s3)
        s4 = [s[i] + dt * k3[i] for i in range(4)]
        k4 = self.get_derivatives(s4)
        
        return [s[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(4)]

    def euler_step(self, s, dt):
        d = self.get_derivatives(s)
        return [s[i] + d[i] * dt for i in range(4)]

    def get_energy(self, s):
        th1, w1, th2, w2 = s
        m1, m2, l1, l2, g = self.params['m1'], self.params['m2'], self.params['l1'], self.params['l2'], self.params['g']
        
        T = 0.5 * (m1 + m2) * l1 * l1 * w1 * w1 \
            + 0.5 * m2 * l2 * l2 * w2 * w2 \
            + m2 * l1 * l2 * w1 * w2 * math.cos(th1 - th2)
            
        y1 = l1 * math.cos(th1)
        y2 = y1 + l2 * math.cos(th2)
        V = -(m1 + m2) * g * y1 - m2 * g * (l2 * math.cos(th2))
        
        return T, V, T + V

    def get_positions(self, s):
        w = self.pen_canvas.winfo_width()
        h = self.pen_canvas.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 600, 600
            
        th1, _, th2, _ = s
        max_l = self.params['l1'] + self.params['l2']
        if max_l == 0: max_l = 1
        scale = (w / 2.5) / max_l
        
        cx, cy = w / 2, h / 3
        
        x1 = cx + self.params['l1'] * scale * math.sin(th1)
        y1 = cy + self.params['l1'] * scale * math.cos(th1)
        
        x2 = x1 + self.params['l2'] * scale * math.sin(th2)
        y2 = y1 + self.params['l2'] * scale * math.cos(th2)
        
        return cx, cy, x1, y1, x2, y2, scale

    def draw_pendulum(self):
        self.pen_canvas.delete("all")
        cx, cy, x1, y1, x2, y2, scale = self.get_positions(self.state)
        
        # Draw trail
        if len(self.trail_data) > 1:
            pts = []
            for p in self.trail_data:
                pts.extend([p[0], p[1]])
            self.pen_canvas.create_line(pts, fill='#aed6f1', width=2, smooth=True)
            
        # Draw rods
        self.pen_canvas.create_line(cx, cy, x1, y1, width=4, fill='#333333', capstyle=tk.ROUND)
        self.pen_canvas.create_line(x1, y1, x2, y2, width=4, fill='#333333', capstyle=tk.ROUND)
        
        # Draw pivot
        self.pen_canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill='#2c3e50', outline='')
        
        # Draw masses
        r1 = max(10, math.sqrt(self.params['m1']) * 8)
        r2 = max(10, math.sqrt(self.params['m2']) * 8)
        
        self.pen_canvas.create_oval(x1-r1, y1-r1, x1+r1, y1+r1, fill='#e74c3c', outline='#c0392b', width=2)
        self.pen_canvas.create_oval(x2-r2, y2-r2, x2+r2, y2+r2, fill='#3498db', outline='#2980b9', width=2)

    def draw_plot(self):
        self.plot_canvas.delete("all")
        if len(self.plot_data) < 2: return
        
        w = self.plot_canvas.winfo_width()
        h = self.plot_canvas.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 800, 250
            
        margin_top, margin_right, margin_bottom, margin_left = 20, 20, 30, 50
        draw_w = w - margin_left - margin_right
        draw_h = h - margin_top - margin_bottom
        
        # Draw axes
        self.plot_canvas.create_line(margin_left, margin_top, margin_left, h - margin_bottom, fill='#aaa')
        self.plot_canvas.create_line(margin_left, h - margin_bottom, w - margin_right, h - margin_bottom, fill='#aaa')
        
        t_min = self.plot_data[0]['t']
        t_max = self.plot_data[-1]['t']
        if t_max - t_min < 5:
            t_max = t_min + 5
            
        def map_x(t_val):
            return margin_left + ((t_val - t_min) / (t_max - t_min)) * draw_w
            
        def map_y(th_val):
            return margin_top + draw_h / 2 - (th_val / 200.0) * (draw_h / 2)
            
        # Draw center line
        self.plot_canvas.create_line(margin_left, map_y(0), w - margin_right, map_y(0), fill='#eee')
        
        for key, color in [('th1', '#e74c3c'), ('th2', '#3498db')]:
            pts = []
            for i in range(len(self.plot_data)):
                d = self.plot_data[i]
                x = map_x(d['t'])
                deg = ((d[key] * 180 / math.pi) % 360 + 360) % 360
                if deg > 180: deg -= 360
                
                y = map_y(deg)
                
                if i > 0:
                    prev_deg = ((self.plot_data[i-1][key] * 180 / math.pi) % 360 + 360) % 360
                    if prev_deg > 180: prev_deg -= 360
                    if abs(deg - prev_deg) > 300:
                        if len(pts) >= 4:
                            self.plot_canvas.create_line(pts, fill=color, width=2)
                        pts = []
                
                pts.extend([x, y])
                
            if len(pts) >= 4:
                self.plot_canvas.create_line(pts, fill=color, width=2)
                
        self.plot_canvas.create_text(margin_left - 5, map_y(180), text="180°", anchor='e', fill='#666')
        self.plot_canvas.create_text(margin_left - 5, map_y(0), text="0°", anchor='e', fill='#666')
        self.plot_canvas.create_text(margin_left - 5, map_y(-180), text="-180°", anchor='e', fill='#666')
        
        self.plot_canvas.create_text(margin_left, h - margin_bottom + 5, text=f"{t_min:.1f}s", anchor='n', fill='#666')
        self.plot_canvas.create_text(w - margin_right, h - margin_bottom + 5, text=f"{t_max:.1f}s", anchor='n', fill='#666')

    def update_stats(self):
        T, V, E = self.get_energy(self.state)
        drift = 0.0
        if self.initial_energy != 0:
            drift = abs((E - self.initial_energy) / self.initial_energy) * 100.0
            
        color_drift = "#00ff00" if drift <= 1 else "#e74c3c"
            
        text = f"时间 (t) : {self.t:.2f} s\n"
        text += f"动能 (T) : {T:.4f} J\n"
        text += f"势能 (V) : {V:.4f} J\n"
        text += f"总能量(E) : {E:.4f} J\n"
        
        # Tkinter Label can't do inline rich text color easily, so we just show it
        text += f"能量漂移 : {drift:.6f} %\n"
        
        if self.params['method'] == 'euler':
            text += "\n欧拉法能量不守恒，漂移会迅速增加！"
            
        self.stats_label.config(text=text)

    def read_params(self):
        try:
            self.params['m1'] = self.vars['m1'].get()
            self.params['m2'] = self.vars['m2'].get()
            self.params['l1'] = self.vars['l1'].get()
            self.params['l2'] = self.vars['l2'].get()
            self.params['g'] = self.vars['g'].get()
            self.params['th1'] = math.radians(self.vars['th1'].get())
            self.params['th2'] = math.radians(self.vars['th2'].get())
            self.params['dt'] = self.vars['dt'].get()
            self.params['method'] = self.method_var.get()
            return True
        except tk.TclError:
            return False

    def on_var_change(self, *args):
        if not self.is_running:
            if self.read_params():
                self.t = 0.0
                self.state = [self.params['th1'], 0.0, self.params['th2'], 0.0]
                self.trail_data = []
                self.plot_data = []
                self.last_record_time = 0.0
                _, _, self.initial_energy = self.get_energy(self.state)
                
                self.draw_pendulum()
                self.draw_plot()
                self.update_stats()

    def reset_simulation(self):
        if self.read_params():
            self.t = 0.0
            self.state = [self.params['th1'], 0.0, self.params['th2'], 0.0]
            self.trail_data = []
            self.plot_data = []
            self.last_record_time = 0.0
            _, _, self.initial_energy = self.get_energy(self.state)
            
            self.draw_pendulum()
            self.draw_plot()
            self.update_stats()

    def toggle_play(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_play.config(text="暂停", bg="#f39c12")
            self.last_frame_time = time.perf_counter()
            self.time_accumulator = 0.0
        else:
            self.btn_play.config(text="开始 / 暂停", bg="#2ecc71")

    def btn_reset_clicked(self):
        self.is_running = False
        self.btn_play.config(text="开始 / 暂停", bg="#2ecc71")
        self.reset_simulation()

    def btn_export_clicked(self):
        if not self.plot_data:
            messagebox.showinfo("提示", "没有可导出的数据，请先运行模拟。")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="double_pendulum_data.csv"
        )
        if filepath:
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time_s", "Theta1_deg", "Theta2_deg"])
                    for d in self.plot_data:
                        d1 = ((d['th1'] * 180 / math.pi) % 360 + 360) % 360
                        d2 = ((d['th2'] * 180 / math.pi) % 360 + 360) % 360
                        if d1 > 180: d1 -= 360
                        if d2 > 180: d2 -= 360
                        writer.writerow([f"{d['t']:.4f}", f"{d1:.4f}", f"{d2:.4f}"])
                messagebox.showinfo("成功", f"数据已导出至:\n{filepath}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败:\n{e}")

    def on_mouse_down(self, event):
        if self.is_running: return
        cx, cy, x1, y1, x2, y2, scale = self.get_positions(self.state)
        
        dist1 = math.hypot(event.x - x1, event.y - y1)
        dist2 = math.hypot(event.x - x2, event.y - y2)
        
        if dist2 < 30:
            self.dragging_node = 2
        elif dist1 < 30:
            self.dragging_node = 1

    def on_mouse_move(self, event):
        if self.dragging_node == 0 or self.is_running: return
        cx, cy, x1, y1, x2, y2, scale = self.get_positions(self.state)
        
        if self.dragging_node == 1:
            dx = event.x - cx
            dy = event.y - cy
            angle = math.atan2(dx, dy)
            self.state[0] = angle
            self.vars['th1'].set(round(math.degrees(angle), 1))
        elif self.dragging_node == 2:
            dx = event.x - x1
            dy = event.y - y1
            angle = math.atan2(dx, dy)
            self.state[2] = angle
            self.vars['th2'].set(round(math.degrees(angle), 1))
            
        self.state[1] = 0.0
        self.state[3] = 0.0
        self.read_params()
        _, _, self.initial_energy = self.get_energy(self.state)
        
        self.draw_pendulum()
        self.update_stats()

    def on_mouse_up(self, event):
        self.dragging_node = 0

    def update_loop(self):
        current_time = time.perf_counter()
        if self.is_running:
            frame_time = current_time - self.last_frame_time
            if frame_time > 0.1: frame_time = 0.1
            
            self.time_accumulator += frame_time
            dt = self.params['dt']
            
            steps = 0
            while self.time_accumulator >= dt and steps < 100: # limit max steps to prevent freeze
                if self.params['method'] == 'rk4':
                    self.state = self.rk4_step(self.state, dt)
                else:
                    self.state = self.euler_step(self.state, dt)
                    
                self.t += dt
                self.time_accumulator -= dt
                steps += 1
                
                if self.t - self.last_record_time >= 0.05:
                    self.plot_data.append({'t': self.t, 'th1': self.state[0], 'th2': self.state[2]})
                    if len(self.plot_data) > self.MAX_PLOT_POINTS:
                        self.plot_data.pop(0)
                        
                    cx, cy, x1, y1, x2, y2, scale = self.get_positions(self.state)
                    self.trail_data.append((x2, y2))
                    if len(self.trail_data) > self.MAX_TRAIL:
                        self.trail_data.pop(0)
                        
                    self.last_record_time = self.t
                    
            self.draw_pendulum()
            self.draw_plot()
            self.update_stats()
            
        self.last_frame_time = current_time
        self.root.after(16, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = DoublePendulumApp(root)
    root.mainloop()
