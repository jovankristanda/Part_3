import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    from scipy.special import iv  # modified Bessel I_v
except Exception:
    iv = None


def main():
    # -------------------------
    # Physical parameters
    # -------------------------
    r0 = 0.01          # m
    R0 = 0.015         # m (baseline)
    D = 1e-5           # m^2/s^2
    k = 1           # 1/s
    c_bulk = 1       # mol/m^3

    # Dimensionless groups
    Phi = r0 * np.sqrt(k / D)
    delta = R0 / r0

    # Grid
    Nl = 40            # l is lamda (>= 30)
    Nt = 60            # t is tetha (>= 30)

    # Plot colormap
    cmap = matplotlib.cm.viridis

    # -------------------------
    # Sanity check A: Phi -> 0 should give psi=1
    # -------------------------
    psi0 = solve_torus_fvm(Nl, Nt, delta=delta, Phi=0.0)
    print("Sanity A (Phi=0): max|psi-1| = ", np.max(np.abs(psi0 - 1.0)))

    print("psi0 min/max:", psi0.min(), psi0.max())

    # -------------------------
    # Baseline solve
    # -------------------------
    psi = solve_torus_fvm(Nl, Nt, delta=delta, Phi=Phi)

    # ---- Sanity check B: bounds of solution ----
    print("psi min/max:", psi.min(), psi.max())

    # Q2: contour/heatmap on cross-section
    plot_cross_section_polar(psi, Nl, Nt, cmap=cmap)

    # Q3: radial line at theta=0 and theta=pi
    lam, theta = grid_centers(Nl, Nt)
    lam_line, psi_t0, psi_tpi = extract_lines(psi, lam, theta)
    plot_lines(lam_line, psi_t0, psi_tpi,
           xlabel=r'$\lambda$ [-]', ylabel=r'$\psi$ [-]',
           title="Q3: radial profiles at θ=0 and θ=π")

    # Q4: compare with sphere and cylinder
    plot_compare_sphere_cyl(lam_line, psi_t0, psi_tpi, Phi)

    # Q5/Q6: sweep R0
    R0_list = np.array([0.015, 0.03, 0.06, 0.12])  # m
    sweep_major_radius(R0_list, r0, Phi, Nl, Nt)


# ---------------------------------------------------------------------
# Geometry & grids
# ---------------------------------------------------------------------
def grid_centers(Nl, Nt):
    dl = 1.0 / Nl
    dth = 2.0 * np.pi / Nt

    lam = np.linspace(dl/2, 1.0 - dl/2, Nl)        # λ_P
    theta = np.linspace(0.0 + dth/2, 2.0*np.pi - dth/2, Nt)  # θ_P (poloidal)
    return lam, theta


def grid_faces(Nl, Nt):
    dl = 1.0 / Nl
    dth = 2.0 * np.pi / Nt

    lam_w = np.linspace(0.0, 1.0 - dl, Nl)         # west face λ_w
    lam_e = lam_w + dl                              # east face λ_e
    th_s  = np.linspace(0.0, 2.0*np.pi - dth, Nt)     # south face θ_s
    th_n = th_s + dth                                # north face θ_n
    return dl, dth, lam_w, lam_e, th_s, th_n


def H(delta, lam, theta):
    return delta + lam * np.cos(theta)


# ---------------------------------------------------------------------
# Assembly: A x = b
# Unknowns: psi[i,j] mapped to p = i*Nt + j
# Radial index i=0..Nl-1, angular index j=0..Nt-1
# BCs:
#   - at λ=1: psi = 1 (Dirichlet)
#   - at λ=0: symmetry dpsi/dλ = 0 (Neumann)
#   - in θ: periodic
# ---------------------------------------------------------------------
def solve_torus_fvm(Nl, Nt, delta, Phi):
    A, b = build_system(Nl, Nt, delta, Phi)

    print("A diag min/max:", np.min(np.diag(A)), np.max(np.diag(A)))
    print("Any NaN in A?", np.isnan(A).any(), "Any inf in A?", np.isinf(A).any())
    print("Any NaN in b?", np.isnan(b).any(), "Any inf in b?", np.isinf(b).any())

    x = np.linalg.solve(A, b)
    return x.reshape((Nl, Nt))



def build_system(Nl, Nt, delta, Phi):
    dl, dth, lam_w, lam_e, th_s, th_n = grid_faces(Nl, Nt)
    lam_c, th_c = grid_centers(Nl, Nt)

    N = Nl * Nt
    A = np.zeros((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i * Nt + j

    # Precompute for speed/readability
    for i in range(Nl):
        for j in range(Nt):
            p = idx(i, j)

            lamP = lam_c[i]
            thP = th_c[j]
            HP = H(delta, lamP, thP)

            # Neighbor indices (theta periodic)
            jN = (j + 1) % Nt
            jS = (j - 1) % Nt

            # Face locations
            lamW = lam_w[i]
            lamE = lam_e[i]
            thS = th_s[j]
            thN = th_n[j]

            # Midpoint-rule face areas and volume (dimensionless, includes 2π)
            # Radial faces:
            Ao = 2.0 * np.pi * lamE * H(delta, lamE, thP) * dth     # at λ=lamE
            Ai = 2.0 * np.pi * lamW * H(delta, lamW, thP) * dth     # at λ=lamW

            # Angular faces (constant theta):
            Accw = 2.0 * np.pi * H(delta, lamP, thN) * dl           # at θ=thN
            Acw  = 2.0 * np.pi * H(delta, lamP, thS) * dl           # at θ=thS

            # Volume:
            Vp = 2.0 * np.pi * lamP * HP * dl * dth

            # Coefficients for interior formula
            aE = Ao / dl
            aW = Ai / dl
            # angular has factor 1/λ_P
            aN = Accw / (lamP * dth)
            aS = Acw  / (lamP * dth)

            aP = aE + aW + aN + aS + (Phi**2) * Vp
            
            
            # -------------------------
            # Radial boundary handling
            # -------------------------
            if i == 0:
                # λ=0 symmetry: no "west" flux.
                # simplest FV: remove W contribution by setting aW=0
                aW = 0.0
                aP = aE + aN + aS + (Phi**2) * Vp

            if i == Nl - 1:
                # Outer boundary at λ=1: psi = 1 (Dirichlet).
                # Treat "east" neighbor as boundary value.
                # Move aE * psi_bc to RHS and remove the neighbor coupling.
                A[p, :] = 0.0
                A[p, p] = 1.0
                b[p] = 1.0
                continue 

            # -------------------------
            # Fill matrix row
            # Equation: aE*psi_E + aW*psi_W + aN*psi_N + aS*psi_S - aP*psi_P = -b
            # We'll store as: (-aP)*P + aE*E + ... = -b  --> multiply by -1:
            # aP*P - aE*E - aW*W - aN*N - aS*S = b
            # This makes diagonal positive, easier to read.
            # -------------------------
            A[p, p] = aP

            # East/West radial neighbors (if exist)
            if i < Nl - 1 and aE != 0.0:
                A[p, idx(i + 1, j)] = -aE
            if i > 0 and aW != 0.0:
                A[p, idx(i - 1, j)] = -aW

            # North/South angular neighbors (periodic always exist)
            A[p, idx(i, jN)] = -aN
            A[p, idx(i, jS)] = -aS

            # RHS already in b[p]
            # (reaction is included in diagonal via Phi^2*Vp)

    return A, b


# ---------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------
def plot_cross_section_polar(psi, Nl, Nt, cmap):
    dl, dth, lam_w, lam_e, th_s, th_n = grid_faces(Nl, Nt)

    r_edges = np.r_[lam_w, 1.0]        # length Nl+1
    th_edges = np.r_[th_s, 2*np.pi]    # length Nt+1

    fig = plt.figure(dpi=144)
    ax = fig.add_subplot(111, projection='polar')

    # C must be (Nl, Nt): rows=r, cols=theta
    pcm = ax.pcolormesh(th_edges, r_edges, psi, shading='auto', cmap=cmap)
    fig.colorbar(pcm, ax=ax, label=r'$\psi$ [-]')

    ax.set_title('Q2: Dimensionless concentration profile (polar)')
    ax.set_rlabel_position(90)
    plt.tight_layout()
    plt.show()


def extract_lines(psi, lam, theta):
    # pick indices closest to theta=0 and theta=pi
    j0 = int(np.argmin(np.abs(theta - 0.0)))
    jpi = int(np.argmin(np.abs((theta - np.pi))))

    psi_t0 = psi[:, j0]
    psi_tpi = psi[:, jpi]
    return lam, psi_t0, psi_tpi


def plot_lines(x, y1, y2, xlabel=r'$\lambda$ [-]', ylabel=r'$\psi$ [-]', title="Profiles"):
    plt.figure(dpi=144)
    plt.plot(x, y1, 'o-', label=r'$\theta=0$')
    plt.plot(x, y2, 'o-', label=r'$\theta=\pi$')
    plt.grid(linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def analytical_sphere(lam, Phi):
    # psi = (1/lam) * sinh(Phi*lam)/sinh(Phi)
    lam_safe = np.maximum(lam, 1e-12)
    return (1.0 / lam_safe) * np.sinh(Phi * lam_safe) / np.sinh(Phi)


def analytical_cylinder(lam, Phi):
    if iv is None:
        raise RuntimeError("scipy not available: cannot compute cylinder analytical solution (I0).")
    return iv(0, Phi * lam) / iv(0, Phi)


def plot_compare_sphere_cyl(lam, psi_t0, psi_tpi, Phi):
    plt.figure(dpi=144)
    plt.plot(lam, psi_t0, 'o', label=r'Torus $\theta=0$')
    plt.plot(lam, psi_tpi, 'o', label=r'Torus $\theta=\pi$')

    ll = np.linspace(1e-6, 1.0, 400)
    plt.plot(ll, analytical_sphere(ll, Phi), '--', label='Sphere (analytical)')

    if iv is not None:
        plt.plot(ll, analytical_cylinder(ll, Phi), '--', label='Cylinder (analytical)')

    plt.gca().invert_xaxis()
    plt.grid(linestyle='--')
    plt.xlabel(r'$\lambda$ [-]')
    plt.ylabel(r'$\psi$ [-]')
    plt.title('Q4: Compare torus vs sphere vs cylinder (dimensionless)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def effectiveness_torus(psi, Nl, Nt, delta):
    dl, dth, _, _, _, _ = grid_faces(Nl, Nt)
    lam, th = grid_centers(Nl, Nt)
    Lam, Th = np.meshgrid(lam, th, indexing='ij')

    Vp = 2.0 * np.pi * Lam * H(delta, Lam, Th) * dl * dth  # dimensionless volume weights
    return np.sum(psi * Vp) / np.sum(Vp)



def effectiveness_sphere(Phi):
    # eta = 3/Phi^2 * (Phi*coth(Phi) - 1)
    if Phi == 0:
        return 1.0
    return 3.0 / (Phi**2) * (Phi / np.tanh(Phi) - 1.0)


def effectiveness_cylinder(Phi):
    # eta = 2/Phi * I1(Phi)/I0(Phi)
    if Phi == 0:
        return 1.0
    if iv is None:
        raise RuntimeError("scipy not available: cannot compute cylinder effectiveness factor.")
    return 2.0 / Phi * (iv(1, Phi) / iv(0, Phi))


def sweep_major_radius(R0_list, r0, Phi, Nl, Nt):
    etas = []
    deltas = []

    for R0 in R0_list:
        delta = R0 / r0
        psi = solve_torus_fvm(Nl, Nt, delta=delta, Phi=Phi)

        lam, th = grid_centers(Nl, Nt)
        lam_line, psi_t0, psi_tpi = extract_lines(psi, lam, th)

        # Q5 dimensionless radial profiles: λ vs ψ
        plt.figure(dpi=144)
        plt.plot(lam_line, psi_t0, 'o-', label=r'$\theta=0$')
        plt.plot(lam_line, psi_tpi, 'o-', label=r'$\theta=\pi$')

        ll = np.linspace(1e-6, 1.0, 400)
        plt.plot(ll, analytical_sphere(ll, Phi), '--', label='Sphere (analytical)')

        if iv is not None:
            plt.plot(ll, analytical_cylinder(ll, Phi), '--', label='Cylinder (analytical)')

        plt.grid(linestyle='--')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$\lambda$ [-]')
        plt.ylabel(r'$\psi$ [-]')
        plt.title(f'Q5: δ={delta:.1f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        eta_t = effectiveness_torus(psi, Nl, Nt, delta)
        etas.append(eta_t)
        deltas.append(delta)

    eta_sph = effectiveness_sphere(Phi)
    eta_cyl = effectiveness_cylinder(Phi) if iv is not None else np.nan

    plt.figure(dpi=144)
    plt.plot(deltas, etas, 'o-', label='Torus (numerical)')
    plt.axhline(eta_sph, linestyle='--', label='Sphere (analytical)')
    if iv is not None:
        plt.axhline(eta_cyl, linestyle='--', label='Cylinder (analytical)')

    plt.grid(linestyle='--')
    plt.xlabel(r'$\delta=R_0/r_0$ [-]')
    plt.ylabel(r'$\eta$ [-]')
    plt.title('Q6: Effectiveness factor vs major radius')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
