import numpy as np
from matplotlib.animation import FuncAnimation
from src.config import *
from src.visuals.plot_utils import compute_green_segment

def create_droplets(ax, positions, alpha=0.7):
    """Create ternary droplets for visualization."""
    try:
        for i, pos in enumerate(positions):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            u, v = np.meshgrid(u, v)
            r = 0.2 * (1 + 0.1 * np.sin(u + v))
            x = pos[0] + r * np.cos(u) * np.sin(v)
            y = pos[1] + r * np.sin(u) * np.sin(v)
            z = pos[2] + r * np.cos(v)
            ax.plot_surface(x, y, z, color=['blue', 'green', 'red'][i % 3], alpha=alpha, edgecolor='none')
    except Exception as e:
        logger.error(f"Create droplets error: {e}")

def demo_greenspline_animation(frames: int = 200) -> None:
    """Generate and save greenspline animation with constant curvature and Boas rendering."""
    try:
        fig = plt.figure(figsize=(10, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(0, 1, 100)
        x, y, nodes = compute_green_segment(t)
        if len(x) == 0:
            logger.error("Empty x array from compute_green_segment, using fallback")
            x, y = np.linspace(0, 1, 100), np.zeros(100)
        z = np.zeros_like(x)
        t_fine = np.linspace(0, 1, len(x))
        kappa = CurvatureVerbismGenerator().curve_map_kappa()
        if len(kappa) != len(x):
            logger.warning(f"Kappa length mismatch ({len(kappa)} vs {len(x)}), resizing kappa")
            kappa = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(kappa)), kappa)
        X_blue, Y_blue, Z_blue, X_gold, Y_gold, Z_gold = create_blob_surface(x, y, z, TUBE_RADIUS, NUM_SIDES, kappa)
        light = LightSource(azdeg=315, altdeg=45)
        rgb_blue = np.ones((X_blue.shape[0], X_blue.shape[1], 3)) * [0, 0, 1]  # Blue
        rgb_gold = np.ones((X_gold.shape[0], X_gold.shape[1], 3)) * [1, 0.84, 0]  # Gold
        shaded_blue = light.shade_rgb(rgb_blue, Z_blue)
        shaded_gold = light.shade_rgb(rgb_gold, Z_gold)
        blob_surface_blue = ax.plot_surface(X_blue, Y_blue, Z_blue, facecolors=shaded_blue, edgecolor='none')
        blob_surface_gold = ax.plot_surface(X_gold, Y_gold, Z_gold, facecolors=shaded_gold, edgecolor='none')
        sph_radius = 1.5
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        u, v = np.meshgrid(u, v)
        x_sph = sph_radius * np.cos(u) * np.sin(v)
        y_sph = sph_radius * np.sin(u) * np.sin(v)
        z_sph = sph_radius * np.cos(v)
        ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.3, rstride=3, cstride=3)
        mersenne_3d = np.array([[0, 0, 1], [0.5, 0, 0], [-0.5, 0, 0]])  # Simplified Mersenne points
        for p in mersenne_3d:
            ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 'r--', alpha=0.5, lw=1)
        printed_frames = set()
        def update(frame):
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            ax.axis('off')
            ax.set_facecolor('black')
            if frame < 50:
                ax.view_init(elev=30, azim=frame * 7.2)
                blob_surface_blue.set_visible(True)
                blob_surface_gold.set_visible(True)
            elif frame < 150:
                local_frame = frame - 50
                swap = np.sin(local_frame / 50 * 2 * np.pi)
                positions = [
                    [0 + swap * 0.5, 0, 0],
                    [0.5 + swap * -0.5, 0, 0],
                    [-0.5 + swap * 0.5, 0, 0]
                ]
                create_droplets(ax, positions, alpha=0.7)
                ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.3, rstride=3, cstride=3)
                for p in mersenne_3d:
                    ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 'r--', alpha=0.5, lw=1)
                ax.text(0, -1.5, 0, 'Ternary Swap: 0/1/e', fontsize=12, color='white', ha='center', va='center')
                blob_surface_blue.set_visible(False)
                blob_surface_gold.set_visible(False)
            else:
                local_frame = frame - 150
                ax.view_init(elev=30, azim=local_frame * 7.2 + 360)
                blob_surface_blue.set_visible(True)
                blob_surface_gold.set_visible(True)
                alpha = 1 - (local_frame / 50)
                if alpha > 0:
                    swap = np.sin((150 + local_frame) / 50 * 2 * np.pi)
                    positions = [[0 + swap * 0.5, 0, 0], [0.5 + swap * -0.5, 0, 0], [-0.5 + swap * 0.5, 0, 0]]
                    create_droplets(ax, positions, alpha=alpha)
                ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.3, rstride=3, cstride=3)
                for p in mersenne_3d:
                    ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 'r--', alpha=0.5 * alpha, lw=1)
            add_light_slicks(ax, alpha=min(1, max(0, (frame - 50) / 100)) if frame < 150 else min(1, max(0, (200 - frame) / 50)))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            if frame % 45 == 0 and frame not in printed_frames:
                kappa_hash = hashlib.sha256((str(np.mean(kappa)) + str(frame)).encode()).hexdigest()[:8]
                print(f"Mnemonic at frame {frame}: {kappa_hash}")
                printed_frames.add(frame)
            return [blob_surface_blue, blob_surface_gold]
        anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        anim.save('greenspline_animation.gif', writer='pillow', fps=20)
        print("Simulating greenspline_animation... Saved as 'greenspline_animation.gif'")
    except Exception as e:
        logger.error(f"Demo greenspline animation error: {e}")

def animate_logo(frames: int = 200) -> None:
    """Generate and save logo animation with ternary swap."""
    try:
        fig, ax = plt.subplots(facecolor='black')
        t = np.linspace(0, 1, 100)
        x, y, _ = compute_green_segment(t)
        if len(x) == 0:
            logger.error("Empty x array from compute_green_segment, using fallback")
            x, y = np.linspace(0, 1, 100), np.zeros(100)
        def update(frame):
            ax.clear()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            ax.set_facecolor('black')
            if frame < 50:
                ax.plot(x + frame * 0.02, y, 'b-', label='Grok Logo')
                ax.scatter([x[0] + frame * 0.02, x[-1] + frame * 0.02], [y[0], y[-1]], c='gold')
            elif frame < 150:
                local_frame = frame - 50
                swap = np.sin(local_frame / 50 * 2 * np.pi)
                pos0 = [0 + swap * 0.5, 0]
                pos1 = [0.5 + swap * -0.5, 0]
                pos2 = [-0.5 + swap * 0.5, 0]
                ax.scatter(pos0[0], pos0[1], c='blue', alpha=0.7, label='0')
                ax.scatter(pos1[0], pos1[1], c='green', alpha=0.7, label='1')
                ax.scatter(pos2[0], pos2[1], c='red', alpha=0.7, label='e')
                ax.text(0, -0.7, 'Ternary Swap', color='white', ha='center', va='center')
            else:
                local_frame = frame - 150
                ax.plot(x + local_frame * 0.02, y, 'b-', label='Grok Logo')
                ax.scatter([x[0] + local_frame * 0.02, x[-1] + local_frame * 0.02], [y[0], y[-1]], c='gold')
                alpha = 1 - (local_frame / 50)
                if alpha > 0:
                    swap = np.sin((150 + local_frame) / 50 * 2 * np.pi)
                    pos0 = [0 + swap * 0.5, 0]
                    pos1 = [0.5 + swap * -0.5, 0]
                    pos2 = [-0.5 + swap * 0.5, 0]
                    ax.scatter(pos0[0], pos0[1], c='blue', alpha=alpha * 0.7)
                    ax.scatter(pos1[0], pos1[1], c='green', alpha=alpha * 0.7)
                    ax.scatter(pos2[0], pos2[1], c='red', alpha=alpha * 0.7)
            ax.legend(loc='upper right')
            return []
        anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        anim.save('logo_animation.gif', writer='pillow', fps=20)
        print("Simulating logo animation... Saved as 'logo_animation.gif'")
    except Exception as e:
        logger.error(f"Animate logo error: {e}")

def demo_shuttle(frames: int = 200) -> None:
    """Generate and save shuttle animation with constant curvature."""
    try:
        fig = plt.figure(figsize=(10, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(0, 1, 100)
        x, y, _ = compute_green_segment(t)
        if len(x) == 0:
            logger.error("Empty x array from compute_green_segment, using fallback")
            x, y = np.linspace(0, 1, 100), np.zeros(100)
        z = ShuttleModel().wave_packet
        t_fine = np.linspace(0, 1, len(x))
        kappa = ShuttleModel().compute_shuttle_kappa(x, y, t_fine)
        if len(kappa) != len(x):
            logger.warning(f"Kappa length mismatch ({len(kappa)} vs {len(x)}), resizing kappa")
            kappa = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(kappa)), kappa)
        X_blue, Y_blue, Z_blue, X_gold, Y_gold, Z_gold = create_blob_surface(x, y, z, TUBE_RADIUS, NUM_SIDES, kappa)
        light = LightSource(azdeg=315, altdeg=45)
        rgb_blue = np.ones((X_blue.shape[0], X_blue.shape[1], 3)) * [0, 0, 1]  # Blue
        rgb_gold = np.ones((X_gold.shape[0], X_gold.shape[1], 3)) * [1, 0.84, 0]  # Gold
        shaded_blue = light.shade_rgb(rgb_blue, Z_blue)
        shaded_gold = light.shade_rgb(rgb_gold, Z_gold)
        blob_surface_blue = ax.plot_surface(X_blue, Y_blue, Z_blue, facecolors=shaded_blue, edgecolor='none')
        blob_surface_gold = ax.plot_surface(X_gold, Y_gold, Z_gold, facecolors=shaded_gold, edgecolor='none')
        def update(frame):
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            ax.axis('off')
            ax.set_facecolor('black')
            offset = frame * 0.01
            X_blue_shifted = X_blue + offset
            X_gold_shifted = X_gold + offset
            ax.plot_surface(X_blue_shifted, Y_blue, Z_blue, facecolors=shaded_blue, edgecolor='none')
            ax.plot_surface(X_gold_shifted, Y_gold, Z_gold, facecolors=shaded_gold, edgecolor='none')
            add_light_slicks(ax)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Shuttle Animation: Frame {frame}")
            if frame % 45 == 0:
                kappa_hash = hashlib.sha256((str(np.mean(kappa)) + str(frame)).encode()).hexdigest()[:8]
                print(f"Mnemonic at frame {frame}: {kappa_hash}")
            return [blob_surface_blue, blob_surface_gold]
        anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        anim.save('shuttle_animation.gif', writer='pillow', fps=20)
        print("Simulating shuttle animation... Saved as 'shuttle_animation.gif'")
    except Exception as e:
        logger.error(f"Demo shuttle error: {e}")
