import numpy as np
import torch
import matplotlib.pyplot as plt
from io_spec import get_rvdatachallenge_dataset, get_dataset_from_template
from ccf import (
    compute_CCFs_mask_gauss,
    compute_CCFs_template_optimized,
    normalize_CCFs,
    compute_CCFs_masked_template,
)
from fitter import CCFFitter, extract_rv_from_fit
from astropy.timeseries import LombScargle
from doppler import shift_spec
import json
import matplotlib.gridspec as gridspec

# Hyperparamètres
verbose: bool = True
wavemin: float = 4000
wavemax: float = None
n_specs = 1000
planetary_signal_amplitudes: list = [10]
planetary_signal_periods: list = [100]
verbose: bool = False
noise_level: float = 0
v_grid_max: int = 20000
v_grid_step: int = 250
v_grid: np.ndarray = np.arange(-v_grid_max, v_grid_max, v_grid_step)
window_size_velocity: int = 820  # Taille de la fenêtre en espace de vitesse en m/s
fit_window_size: int = 10000  # Taille de la fenêtre pour le fit en m/s
mask_type: str = "G2"  # Type de masque à utiliser (G2, HARPN_Kitcat, ESPRESSO_F9)
anomalous_spectra: list[int] = [334, 464]


def main():
    """
    Fonction principale pour exécuter le script de calcul des CCFs et des RVs.
    Cette fonction génère un dataset, calcule les CCFs avec un masque et un template,
    normalise les CCFs, effectue le fit des CCFs et analyse les RVs avec Lomb-Scargle.
    """

    print("=== EXPÉRIENCE RV DATA CHALLENGE ===")
    torch.cuda.empty_cache()

    print("Génération du dataset...")

    dataset, wavegrid, template, jdb = get_rvdatachallenge_dataset(
        n_specs=n_specs,
        wavemin=wavemin,
        wavemax=wavemax,
        planetary_signal_amplitudes=planetary_signal_amplitudes,
        planetary_signal_periods=planetary_signal_periods,
        verbose=verbose,
        noise_level=noise_level,
    )

    # template = np.load(
    #     "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/spec.npy"
    # )
    # template = template / np.max(template)  # Normalisation du template
    # wavegrid = np.load(
    #     "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/wavegrid.npy"
    # )

    # velocities = np.zeros(n_specs, dtype=np.float32)
    # for amplitude, period in zip(planetary_signal_amplitudes, planetary_signal_periods):
    #     velocities += amplitude * np.sin(2 * np.pi * np.arange(n_specs) / period)

    # dataset, wavegrid, template = get_dataset_from_template(
    #     template=template,
    #     wavegrid=wavegrid,
    #     velocities=velocities,
    #     wavemin=wavemin,
    #     wavemax=wavemax,
    #     noise_level=noise_level,
    #     dtype=torch.float32,
    #     verbose=True,
    # )
    # print("Dataset généré avec succès!")

    print("Calcul des CCFs avec masque...")

    CCFs_mask = compute_CCFs_mask_gauss(
        specs=dataset,
        v_grid=v_grid,
        wavegrid=wavegrid,
        mask_type=mask_type,  # Type de masque à utiliser (G2, HARPN_Kitcat, ESPRESSO_F9)
        window_size_velocity=window_size_velocity,  # Taille de la fenêtre en espace de vitesse
        verbose=verbose,
    )
    print("Calcul des CCFs pour le template...")
    CCFs_template = compute_CCFs_template_optimized(
        specs=dataset,
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        dtype=torch.float32,
        verbose=verbose,
    )
    print("Normalisation des CCFs...")

    CCFs_mask = normalize_CCFs(CCFs_mask)
    CCFs_template = normalize_CCFs(CCFs_template)

    print("CCFs calculés avec succès!")

    # Tracé des CCFs
    random_index = np.random.randint(0, len(CCFs_mask))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        v_grid,
        CCFs_mask[random_index],
        label=f"Random CCF Mask {random_index}",
        color="blue",
    )
    plt.title("CCF avec Masque")
    plt.xlabel("Vitesse (m/s)")
    plt.ylabel("CCF")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(
        v_grid,
        CCFs_template[random_index],
        label=f"Random CCF Template {random_index}",
        color="orange",
    )
    plt.title("CCF avec Template")
    plt.xlabel("Vitesse (m/s)")
    plt.ylabel("CCF")
    plt.legend()
    plt.grid(True)
    plt.suptitle("Comparaison des CCFs")
    plt.tight_layout()
    plt.show()

    print("Fit des CCFs...")

    fitter = CCFFitter()

    rvs_mask = []
    rvs_template = []
    N_specs = len(dataset)

    for i in range(N_specs):
        res_mask = fitter.fit_ccf(
            CCFs_mask[i],
            v_grid,
            window_size=fit_window_size,
            orientation="down",
            model="gaussian",
        )

        if i == random_index:
            fitter.plot_fit_result(res_mask)

        rvs_mask.append(extract_rv_from_fit(res_mask)[0])

        res_template = fitter.fit_ccf(
            CCFs_template[i],
            v_grid,
            window_size=fit_window_size,
            orientation="up",
            model="gaussian",
        )

        if i == random_index:
            fitter.plot_fit_result(res_template)

        rvs_template.append(extract_rv_from_fit(res_template)[0])

    # Création des tableaux pour les vitesses radiales
    rvs_mask = np.array(rvs_mask)
    rvs_template = np.array(rvs_template)

    # Plots des rvs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(jdb, rvs_mask, label="RV Mask", color="blue")
    plt.title("Vitesse Radiale avec Masque")
    plt.xlabel("Temps (jours juliens)")
    plt.ylabel("Vitesse Radiale (m/s)")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(jdb, rvs_template, label="RV Template", color="orange")
    plt.title("Vitesse Radiale avec Template")
    plt.xlabel("Temps (jours juliens)")
    plt.ylabel("Vitesse Radiale (m/s)")
    plt.grid(True)
    plt.legend()
    plt.suptitle("Comparaison des Vitesses Radiales")
    plt.show()

    # Analyse Lomb-Scargle pour les RVs

    # Calcul du Lomb-Scargle pour les RVs
    frequency_mask, power_mask = LombScargle(jdb, rvs_mask).autopower(
        samples_per_peak=100
    )
    frequency_template, power_template = LombScargle(jdb, rvs_template).autopower(
        samples_per_peak=100
    )

    period_mask = 1 / frequency_mask
    period_template = 1 / frequency_template

    # Définir la fraction de zoom (±80% autour de chaque période)
    zoom_fraction = 0.8

    # Boucle sur chaque période injectée
    for P in planetary_signal_periods:
        window = P * zoom_fraction
        fig, ax = plt.subplots(figsize=(8, 4))

        # Tracé des periodogrammes Mask et Template
        ax.plot(period_mask, power_mask, drawstyle="steps-mid", label="Mask")
        ax.plot(
            period_template,
            power_template,
            drawstyle="steps-mid",
            color="orange",
            label="Template",
        )

        # Ligne verticale de la période injectée
        # ax.axvline(P, linestyle="--", color="red", label=f"{P:.2f} j")

        # Zoom autour de la période
        # ax.set_xlim(P - window, P + window)

        # Étiquettes et titre
        ax.set_title(f"Zoom Lomb–Scargle autour de {P:.2f} jours")
        ax.set_xlabel("Période (jours)")
        ax.set_ylabel("Puissance")

        # Légende et grille
        ax.legend()
        ax.grid(True)

        # Affichage
        plt.tight_layout()
        plt.show()

    # planetary_signal = np.zeros(n_specs, dtype=np.float32)
    # for amplitude, period in zip(planetary_signal_amplitudes, planetary_signal_periods):
    #     planetary_signal += amplitude * np.sin(2 * np.pi * np.arange(n_specs) / period)
    # # Plot rv vraies vs rv estimées
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(
    #     planetary_signal,
    #     rvs_mask,
    #     label="RV Mask",
    #     color="blue",
    # )
    # plt.title("Vin vs Vout (Méthode Masque) (m/s)")
    # plt.xlabel("Vitesse Réelle")
    # plt.ylabel("Vitesse prédite")
    # plt.grid(True)
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(
    #     planetary_signal,
    #     rvs_template,
    #     label="RV Template",
    #     color="orange",
    # )
    # plt.title("Vin vs Vout (Méthode Template) (m/s)")
    # plt.xlabel("Vitesse Réelle")
    # plt.ylabel("Vitesse prédite")
    # plt.grid(True)
    # plt.legend()
    # plt.suptitle("Comparaison des Vitesses Radiales avec Périodes Injectées")
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(period_mask, power_mask, label="Lomb-Scargle Mask", color="blue")
    # plt.title("Lomb-Scargle pour RV Mask")
    # plt.xlabel("Période (s)")
    # plt.ylabel("Puissance")
    # plt.xlim(P - 40, P + 40)  # Zoom sur la période attendue
    # plt.ylim(-0.1, 1.1)  # Ajustement de l'axe y pour une meilleure visibilité
    # plt.grid(True)
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(period_template, power_template, label="Lomb-Scargle Template", color="orange")
    # plt.title("Lomb-Scargle pour RV Template")
    # plt.xlabel("Période (s)")
    # plt.ylabel("Puissance")
    # plt.xlim(P - 40, P + 40)  # Zoom sur la période attendue
    # plt.ylim(-0.1, 1.1)  # Ajustement de l'axe y pour une meilleure visibilité

    # plt.grid(True)
    # plt.legend()
    # plt.suptitle("Analyse Lomb-Scargle des Vitesses Radiales")
    # plt.tight_layout()
    # plt.show()


def exp1():
    """
    Fonction d'expérience 1 pour tester l'évolution du pic dans le périodo pour plusieurs Kp à la même période.
    Objectif : Voir si+ l'amplitude du signal est grande plus le pic est grand et plus il est proche de la période attendue.
    """
    print("=== EXPÉRIENCE 1 ===")

    planetary_signal_amplitudes = [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    planetary_signal_periods = [10]

    for amplitude in planetary_signal_amplitudes:
        print(f"Amplitude du signal planétaire : {amplitude} m/s")

        torch.cuda.empty_cache()

        print("Génération du dataset...")

        dataset, wavegrid, template, jdb = get_rvdatachallenge_dataset(
            n_specs=n_specs,
            wavemin=wavemin,
            wavemax=wavemax,
            planetary_signal_amplitudes=[amplitude],
            planetary_signal_periods=planetary_signal_periods,
            verbose=verbose,
            noise_level=noise_level,
        )
        if np.max(anomalous_spectra) < len(dataset) - 1:
            dataset = np.delete(dataset, anomalous_spectra, axis=0)
            jdb = np.delete(jdb, anomalous_spectra, axis=0)

        print("Calcul des CCFs avec masque...")

        CCFs_mask = compute_CCFs_mask_gauss(
            specs=dataset,
            v_grid=v_grid,
            wavegrid=wavegrid,
            mask_type=mask_type,  # Type de masque à utiliser (G2, HARPN_Kitcat, ESPRESSO_F9)
            window_size_velocity=window_size_velocity,  # Taille de la fenêtre en espace de vitesse
            verbose=verbose,
        )

        print("Calcul des CCFs pour le template...")
        CCFs_template = compute_CCFs_template_optimized(
            specs=dataset,
            template=template,
            wavegrid=wavegrid,
            v_grid=v_grid,
            dtype=torch.float32,
            verbose=verbose,
        )

        print("Calcul des CCFs pour le template masqué...")
        CCFs_masked_template = compute_CCFs_masked_template(
            specs=dataset,
            template=template,
            wavegrid=wavegrid,
            v_grid=v_grid,
            mask_type=mask_type,  # Type de masque à utiliser (G2, HARPN_Kitcat, ESPRESSO_F9)
            window_size_velocity=window_size_velocity,  # Taille de la fenêtre en espace de vitesse
            verbose=verbose,
        )
        print("Normalisation des CCFs...")

        CCFs_mask = normalize_CCFs(CCFs_mask)
        CCFs_template = normalize_CCFs(CCFs_template)
        CCFs_masked_template = normalize_CCFs(CCFs_masked_template)

        print("CCFs calculés avec succès!")

        print("Calcul des bissecteurs...")

        # Tracé des CCFs + bissector

        print("Fit des CCFs...")

        fitter = CCFFitter()

        rvs_mask = []
        rvs_template = []
        rvs_masked_template = []
        N_specs = len(dataset)
        random_index = np.random.randint(N_specs)
        for i in range(N_specs):
            res_mask = fitter.fit_ccf(
                CCFs_mask[i],
                v_grid,
                window_size=fit_window_size,
                orientation="down",
                model="gaussian",
            )

            # bissector_mask = fitter.compute_bisector(
            #     CCFs_mask[i],
            #     v_grid,
            #     20000,
            #     "down",
            # )

            # if i == random_index:
            #     fitter.plot_bisector(bissector_mask, show_levels=False)

            rvs_mask.append(extract_rv_from_fit(res_mask)[0])

            res_template = fitter.fit_ccf(
                CCFs_template[i],
                v_grid,
                window_size=20000,
                orientation="up",
                model="gaussian",
            )

            # bissector_template = fitter.compute_bisector(
            #     CCFs_template[i], v_grid, 20000, "up"
            # )

            # if i == random_index:
            #     fitter.plot_bisector(bissector_template, show_levels=False)

            rvs_template.append(extract_rv_from_fit(res_template)[0])

            res_masked_template = fitter.fit_ccf(
                CCFs_masked_template[i],
                v_grid,
                window_size=20000,
                orientation="down",
                model="gaussian",
            )

            # bissector_masked_template = fitter.compute_bisector(
            #     CCFs_masked_template[i], v_grid, 10000, "down"
            # )

            # if i == random_index:
            #     fitter.plot_bisector(bissector_masked_template, show_levels=False)

            rvs_masked_template.append(extract_rv_from_fit(res_masked_template)[0])

        # Création des tableaux pour les vitesses radiales
        rvs_mask = np.array(rvs_mask)
        rvs_template = np.array(rvs_template)
        rvs_masked_template = np.array(rvs_masked_template)

        # # Plots des rvs avec des marqueurs visibles
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 3, 1)
        # plt.plot(
        #     jdb,
        #     rvs_mask,
        #     marker="o",
        #     linestyle="",
        #     color="blue",
        #     label="RV Mask",
        #     markersize=5,
        # )
        # plt.title("Vitesse Radiale avec Masque")
        # plt.xlabel("Temps (jours juliens)")
        # plt.ylabel("Vitesse Radiale (m/s)")
        # plt.grid(True)
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(
        #     jdb,
        #     rvs_template,
        #     marker="x",
        #     linestyle="",
        #     color="orange",
        #     label="RV Template",
        #     markersize=5,
        # )
        # plt.title("Vitesse Radiale avec Template")
        # plt.xlabel("Temps (jours juliens)")
        # plt.ylabel("Vitesse Radiale (m/s)")
        # plt.grid(True)
        # plt.legend()
        # plt.subplot(1, 3, 3)
        # plt.plot(
        #     jdb,
        #     rvs_masked_template,
        #     marker="s",
        #     linestyle="",
        #     color="green",
        #     label="RV Masked Template",
        #     markersize=5,
        # )
        # plt.title("Vitesse Radiale avec Template Masqué")
        # plt.xlabel("Temps (jours juliens)")
        # plt.ylabel("Vitesse Radiale (m/s)")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.suptitle("Comparaison des Vitesses Radiales")
        # plt.savefig(
        #     f"/home/tliopis/Codes/lagrange_llopis_mary_2025/results/ccf/exp1/rvs_amplitude_{amplitude}.png",
        #     dpi=300,
        # )
        # plt.close()  # Ferme la figure pour libérer la mémoire

        # Analyse Lomb-Scargle pour les RVs

        # Calcul du Lomb-Scargle pour les RVs
        frequency_mask, power_mask = LombScargle(jdb, rvs_mask).autopower(
            samples_per_peak=100
        )
        frequency_template, power_template = LombScargle(jdb, rvs_template).autopower(
            samples_per_peak=100
        )

        frequency_masked_template, power_masked_template = LombScargle(
            jdb, rvs_masked_template
        ).autopower(samples_per_peak=100)

        period_mask = 1 / frequency_mask
        period_template = 1 / frequency_template
        period_masked_template = 1 / frequency_masked_template

        peak_mask = get_peak_value(
            power_mask,
            period_mask,
            tolerance=3,
            expected_period_for_peak=planetary_signal_periods[0],
        )

        peak_template = get_peak_value(
            power_template,
            period_template,
            tolerance=3,
            expected_period_for_peak=planetary_signal_periods[0],
        )

        peak_masked_template = get_peak_value(
            power_masked_template,
            period_masked_template,
            tolerance=3,
            expected_period_for_peak=planetary_signal_periods[0],
        )

        # Boucle sur chaque période injectée
        for P in planetary_signal_periods:
            fig, ax = plt.subplots(figsize=(8, 4))

            # Tracé des periodogrammes Mask et Template
            ax.plot(period_mask, power_mask, drawstyle="steps-mid", label="Mask")
            ax.plot(
                period_template,
                power_template,
                drawstyle="steps-mid",
                color="orange",
                label="Template",
            )
            ax.plot(
                period_masked_template,
                power_masked_template,
                drawstyle="steps-mid",
                color="green",
                label="Masked Template",
            )
            # Ligne verticale de la période injectée
            # ax.axvline(P, linestyle="--", color="red", label=f"{P:.2f} j")

            # Zoom autour de la période
            ax.set_xlim(P - 3, P + 3)
            ax.set_ylim(-0.03, 0.5)

            # Étiquettes et titre
            ax.set_title(
                f"Zoom Lomb–Scargle pour signal d'amplitude {amplitude} et période {planetary_signal_periods[0]} jours"
            )
            ax.set_xlabel("Période (jours)")
            ax.set_ylabel("Puissance")

            # Légende et grille
            ax.legend()
            ax.grid(True)

            # Affichage
            plt.tight_layout()
            plt.figtext(
                0.3,
                0.8,
                f"Amplitude: {amplitude} m/s\nPeak mask: {peak_mask:.4f}\nPeak template: {peak_template:.4f}\nPeak masked template: {peak_masked_template:.4f}",
                ha="center",
                fontsize=12,
                backgroundcolor="lightyellow",
            )
            plt.savefig(
                f"/home/tliopis/Codes/lagrange_llopis_mary_2025/results/ccf/exp1/lomb_scargle_amplitude_{amplitude}_period_{planetary_signal_periods[0]}.png",
                dpi=300,
            )
            plt.close()  # Affiche la figure


def get_peak_value(periodogram, period_tolerance, expected_period_for_peak):
    power_vector, frequency_vector = periodogram

    # Conversion des fréquences en périodes
    period_vector = 1 / frequency_vector

    # Fenêtre autour de la période attendue
    period_range = (
        expected_period_for_peak - period_tolerance,
        expected_period_for_peak + period_tolerance,
    )

    mask = (period_vector >= period_range[0]) & (period_vector <= period_range[1])

    if np.any(mask):
        local_powers = power_vector[mask]
        local_periods = period_vector[mask]
        idx_max = np.argmax(local_powers)
        peak_power = local_powers[idx_max]
        peak_period = local_periods[idx_max]
    else:
        peak_power = 0
        peak_period = None

    # Vérifier si le pic est le pic principal du périodogramme càd c'est le pic le plus grand dup
    is_peak_main = peak_power == np.max(power_vector)

    return peak_power, peak_period, int(is_peak_main)


def get_ccfs(
    specs,
    v_grid,
    wavegrid,
    mask_type,
    template,
    window_size_velocity,
    verbose=False,
):
    print("Calcul des CCFs avec masque...")

    CCFs_mask = compute_CCFs_mask_gauss(
        specs=specs,
        v_grid=v_grid,
        wavegrid=wavegrid,
        mask_type=mask_type,
        window_size_velocity=window_size_velocity,
        verbose=verbose,
    )

    print("Calcul des CCFs pour le template...")
    CCFs_template = compute_CCFs_template_optimized(
        specs=specs,
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        dtype=torch.float32,
        verbose=verbose,
    )

    print("Calcul des CCFs pour le template masqué...")
    CCFs_masked_template = compute_CCFs_masked_template(
        specs=specs,
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        mask_type=mask_type,
        window_size_velocity=window_size_velocity,
        verbose=verbose,
    )
    print("Normalisation des CCFs...")

    CCFs_mask = normalize_CCFs(CCFs_mask)
    CCFs_template = normalize_CCFs(CCFs_template)
    CCFs_masked_template = normalize_CCFs(CCFs_masked_template)

    return CCFs_mask, CCFs_template, CCFs_masked_template


def get_rvs(CCFs_mask, CCFs_template, CCFs_masked_template, fit_window_size):
    rvs_mask = []
    rvs_template = []
    rvs_masked_template = []
    N_specs = CCFs_mask.shape[0]

    fitter = CCFFitter()

    random_index = np.random.randint(N_specs)
    for i in range(N_specs):
        res_mask = fitter.fit_ccf(
            CCFs_mask[i],
            v_grid,
            window_size=fit_window_size,
            orientation="down",
            model="gaussian",
        )

        # bissector_mask = fitter.compute_bisector(
        #     CCFs_mask[i],
        #     v_grid,
        #     20000,
        #     "down",
        # )

        # if i == random_index:
        #     fitter.plot_bisector(bissector_mask, show_levels=False)

        rvs_mask.append(extract_rv_from_fit(res_mask)[0])

        res_template = fitter.fit_ccf(
            CCFs_template[i],
            v_grid,
            window_size=fit_window_size,
            orientation="up",
            model="gaussian",
        )

        # bissector_template = fitter.compute_bisector(
        #     CCFs_template[i], v_grid, 20000, "up"
        # )

        # if i == random_index:
        #     fitter.plot_bisector(bissector_template, show_levels=False)

        rvs_template.append(extract_rv_from_fit(res_template)[0])

        res_masked_template = fitter.fit_ccf(
            CCFs_masked_template[i],
            v_grid,
            window_size=fit_window_size,
            orientation="down",
            model="gaussian",
        )

        # bissector_masked_template = fitter.compute_bisector(
        #     CCFs_masked_template[i], v_grid, 10000, "down"
        # )

        # if i == random_index:
        #     fitter.plot_bisector(bissector_masked_template, show_levels=False)

        rvs_masked_template.append(extract_rv_from_fit(res_masked_template)[0])

    # Création des tableaux pour les vitesses radiales
    rvs_mask = np.array(rvs_mask)
    rvs_template = np.array(rvs_template)
    rvs_masked_template = np.array(rvs_masked_template)

    return rvs_mask, rvs_template, rvs_masked_template


def get_periodograms(rvs_mask, rvs_template, rvs_masked_template, time_vector):
    frequency_mask, power_mask = LombScargle(time_vector, rvs_mask).autopower(
        samples_per_peak=100
    )
    frequency_template, power_template = LombScargle(
        time_vector, rvs_template
    ).autopower(samples_per_peak=100)

    frequency_masked_template, power_masked_template = LombScargle(
        time_vector, rvs_masked_template
    ).autopower(samples_per_peak=100)

    periodogram_mask = (power_mask, frequency_mask)
    periodogram_template = (power_template, frequency_template)
    periodogram_masked_template = (power_masked_template, frequency_masked_template)

    return periodogram_mask, periodogram_template, periodogram_masked_template


def exp2():
    """
    Soit un signal Kp * sin(t * 2pi/P) d'amplitude Kp et de période P
    Ce teste vise à calculer les rvs par CCF d'un jeu de donnée pour un ensemble de signaux (Kp, P) (un à la fois) et de calculer pour chaque méthode de CCF l'amplitude du pic associé à la période
    """
    if verbose:
        print("Récupérer le dataset...")

    dataset, wavegrid, template, jdb = get_rvdatachallenge_dataset(
        n_specs=1000,
        wavemin=4000,
        wavemax=None,
        verbose=verbose,
        noise_level=0,
    )
    if np.max(anomalous_spectra) < len(dataset) - 1:
        dataset = np.delete(dataset, anomalous_spectra, axis=0)
        jdb = np.delete(jdb, anomalous_spectra, axis=0)

    amplitudes = [0.01, 0.1, 1, 10, 100]
    periods = [10, 100, 1000]

    results = {}
    for P in periods:
        for Kp in amplitudes:
            if verbose:
                print(f"Injection du signal (Kp, P) = ({Kp}/{P})...")
            shifted_dataset = []
            for i in range(len(dataset)):
                if verbose:
                    print(f"Traitement du spectre {i + 1}/{n_specs}")

                velocity = Kp * np.sin(2 * np.pi * (jdb[i] / P))

                shifted_spec = shift_spec(
                    spec=dataset[i],
                    wavegrid=wavegrid,
                    velocities=np.array([velocity]),
                    dtype=torch.float64,
                )

                shifted_spec = shifted_spec.squeeze(0).cpu().numpy()
                shifted_dataset.append(shifted_spec)

            shifted_dataset = np.array(shifted_dataset)

            CCFs_mask, CCFs_template, CCFs_masked_template = get_ccfs(
                specs=shifted_dataset,
                v_grid=v_grid,
                wavegrid=wavegrid,
                window_size_velocity=window_size_velocity,
                template=template,
                mask_type="G2",
                verbose=verbose,
            )

            rvs_mask, rvs_template, rvs_masked_template = get_rvs(
                CCFs_mask, CCFs_template, CCFs_masked_template, fit_window_size
            )

            periodo_mask, periodo_template, periodo_masked_template = get_periodograms(
                rvs_mask, rvs_template, rvs_masked_template, time_vector=jdb
            )

            perfs_mask = get_peak_value(
                periodo_mask, period_tolerance=3, expected_period_for_peak=P
            )

            perfs_template = get_peak_value(
                periodo_template, period_tolerance=3, expected_period_for_peak=P
            )

            perfs_masked_template = get_peak_value(
                periodo_masked_template, period_tolerance=3, expected_period_for_peak=P
            )

            print("=== Perfs ===")
            print(f"(Kp, P) = ({Kp} / {P})")
            print("Mask : ", perfs_mask)
            print("Template : ", perfs_template)
            print("Masked Template : ", perfs_masked_template)

            # Assure que les niveaux "mask", "template", "masked_template" existent
            for key in ["mask", "template", "masked_template"]:
                if key not in results:
                    results[key] = {}
                if Kp not in results[key]:
                    results[key][Kp] = {}

            # Attribution des valeurs
            results["mask"][Kp][P] = perfs_mask
            results["template"][Kp][P] = perfs_template
            results["masked_template"][Kp][P] = perfs_masked_template

            with open(
                "/home/tliopis/Codes/lagrange_llopis_mary_2025/results/ccf/exp2/res.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results, f)

            del (
                shifted_dataset,
                CCFs_mask,
                CCFs_template,
                CCFs_masked_template,
                rvs_mask,
                rvs_template,
                rvs_masked_template,
                periodo_mask,
                periodo_template,
                perfs_mask,
                perfs_template,
                perfs_masked_template,
            )


def plot_exp2():
    # Chargement du fichier JSON
    with open("results/ccf/exp2/res.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    methods = ["mask", "template", "masked_template"]
    all_Kps = sorted(
        {Kp for m in methods for Kp in results[m].keys()}, key=lambda x: float(x)
    )
    all_Ps = sorted(
        {P for m in methods for Kp in results[m].values() for P in Kp.keys()},
        key=lambda x: float(x),
    )

    # === PREMIER GRAPHIQUE: Peak Power ===
    # Figure + grille manuelle avec espace pour colorbar
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    for i, method in enumerate(methods):
        data = results[method]
        power_matrix = np.full((len(all_Kps), len(all_Ps)), np.nan)
        period_matrix = np.full((len(all_Kps), len(all_Ps)), np.nan)

        for y, Kp in enumerate(all_Kps):
            for x, P in enumerate(all_Ps):
                if Kp in data and P in data[Kp]:
                    power, period, _ = data[Kp][P]  # Ignore is_main_peak ici
                    power_matrix[y, x] = power
                    period_matrix[y, x] = period

        ax = axs[i]
        im = ax.imshow(
            power_matrix,
            origin="lower",
            cmap="viridis",
            extent=[0, len(all_Ps), 0, len(all_Kps)],
            aspect="auto",
        )

        for y in range(len(all_Kps)):
            for x in range(len(all_Ps)):
                val = period_matrix[y, x]
                if not np.isnan(val):
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        ax.set_title(method)
        ax.set_xticks(np.arange(len(all_Ps)) + 0.5)
        ax.set_xticklabels([str(p) for p in all_Ps])
        ax.set_yticks(np.arange(len(all_Kps)) + 0.5)
        ax.set_yticklabels([str(k) for k in all_Kps])
        ax.set_xlabel("Période injectée [j]")
        if i == 0:
            ax.set_ylabel("Kp injecté [m/s]")

    # Colorbar proprement placée à droite
    cax = fig.add_subplot(gs[3])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Peak power")

    plt.suptitle("Détection du pic pour chaque méthode", fontsize=14)
    plt.savefig("results/ccf/exp2/peak_power_comparison.png", dpi=300)
    plt.show()

    # === DEUXIÈME GRAPHIQUE: Is Main Peak ===
    # Figure pour le graphique is_main_peak
    fig2 = plt.figure(figsize=(18, 6))
    gs2 = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    axs2 = [fig2.add_subplot(gs2[i]) for i in range(3)]

    # Colormap personnalisée: Rouge (0) = pas le pic principal, Vert (1) = pic principal
    from matplotlib.colors import ListedColormap

    colors = ["red", "green"]
    custom_cmap = ListedColormap(colors)

    for i, method in enumerate(methods):
        data = results[method]
        is_main_matrix = np.full((len(all_Kps), len(all_Ps)), np.nan)
        power_matrix = np.full((len(all_Kps), len(all_Ps)), np.nan)

        for y, Kp in enumerate(all_Kps):
            for x, P in enumerate(all_Ps):
                if Kp in data and P in data[Kp]:
                    power, period, is_main = data[Kp][P]
                    is_main_matrix[y, x] = int(is_main)  # 0 ou 1
                    power_matrix[y, x] = power

        ax = axs2[i]
        im2 = ax.imshow(
            is_main_matrix,
            origin="lower",
            cmap=custom_cmap,
            extent=[0, len(all_Ps), 0, len(all_Kps)],
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        # Afficher la puissance du pic sur chaque case
        for y in range(len(all_Kps)):
            for x in range(len(all_Ps)):
                power_val = power_matrix[y, x]
                is_main_val = is_main_matrix[y, x]
                if not np.isnan(power_val):
                    # Couleur du texte: blanc sur rouge, noir sur vert
                    text_color = "white" if is_main_val == 0 else "black"
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        f"{power_val:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        weight="bold",
                    )

        ax.set_title(f"{method} - Is Main Peak")
        ax.set_xticks(np.arange(len(all_Ps)) + 0.5)
        ax.set_xticklabels([str(p) for p in all_Ps])
        ax.set_yticks(np.arange(len(all_Kps)) + 0.5)
        ax.set_yticklabels([str(k) for k in all_Kps])
        ax.set_xlabel("Période injectée [j]")
        if i == 0:
            ax.set_ylabel("Kp injecté [m/s]")

    # Colorbar pour le graphique is_main_peak
    cax2 = fig2.add_subplot(gs2[3])
    cbar2 = fig2.colorbar(im2, cax=cax2, ticks=[0, 1])
    cbar2.set_label("Is Main Peak")
    cbar2.set_ticklabels(["Non (Rouge)", "Oui (Vert)"])

    plt.suptitle("Pic principal détecté (Vert) vs Pic secondaire (Rouge)", fontsize=14)
    plt.savefig("results/ccf/exp2/is_main_peak_comparison.png", dpi=300)
    plt.show()


def exp3():
    """
    Expérience 3: Plot des périodogrammes pour Kp = 1 m/s et P = 100 jours
    sur une gamme de périodes de 1 à 1100 jours pour analyser les performances
    de détection en fonction de la période analysée.
    """
    print("=== EXPÉRIENCE 3 ===")
    print("Analyse des périodogrammes avec signal fixe (Kp=1 m/s, P=100j)")

    # Paramètres de l'expérience
    signal_amplitude = 1.0  # m/s
    signal_period = 100  # jours

    # Générer le dataset avec le signal planétaire fixe
    print("Génération du dataset avec signal planétaire...")
    dataset, wavegrid, template, jdb = get_rvdatachallenge_dataset(
        n_specs=n_specs,
        wavemin=wavemin,
        wavemax=wavemax,
        planetary_signal_amplitudes=[signal_amplitude],
        planetary_signal_periods=[signal_period],
        verbose=verbose,
        noise_level=noise_level,
    )

    # Supprimer les spectres anomalous si nécessaire
    if np.max(anomalous_spectra) < len(dataset) - 1:
        dataset = np.delete(dataset, anomalous_spectra, axis=0)
        jdb = np.delete(jdb, anomalous_spectra, axis=0)

    print("Calcul des CCFs...")
    # Calculer les CCFs pour les trois méthodes
    CCFs_mask, CCFs_template, CCFs_masked_template = get_ccfs(
        specs=dataset,
        v_grid=v_grid,
        wavegrid=wavegrid,
        window_size_velocity=window_size_velocity,
        template=template,
        mask_type=mask_type,
        verbose=verbose,
    )

    print("Extraction des vitesses radiales...")
    # Extraire les vitesses radiales
    rvs_mask, rvs_template, rvs_masked_template = get_rvs(
        CCFs_mask, CCFs_template, CCFs_masked_template, fit_window_size
    )

    print("Calcul des périodogrammes...")
    # Calculer les périodogrammes
    periodo_mask, periodo_template, periodo_masked_template = get_periodograms(
        rvs_mask, rvs_template, rvs_masked_template, time_vector=jdb
    )

    # Extraire les données des périodogrammes
    power_mask, frequency_mask = periodo_mask
    power_template, frequency_template = periodo_template
    power_masked_template, frequency_masked_template = periodo_masked_template

    # Convertir les fréquences en périodes
    period_mask = 1 / frequency_mask
    period_template = 1 / frequency_template
    period_masked_template = 1 / frequency_masked_template

    print("Création des graphiques...")

    # === GRAPHIQUE 1: Périodogrammes complets ===
    fig1, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Limites pour le zoom sur la période d'intérêt
    period_min, period_max = 1, 1100

    # Périodogramme Mask
    mask_in_range = (period_mask >= period_min) & (period_mask <= period_max)
    axes[0].plot(
        period_mask[mask_in_range],
        power_mask[mask_in_range],
        "b-",
        linewidth=1.5,
        label="Mask",
    )
    axes[0].axvline(
        signal_period,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Signal injecté ({signal_period}j)",
    )
    axes[0].set_title("Périodogramme - Méthode Mask")
    axes[0].set_ylabel("Puissance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(period_min, period_max)
    axes[0].set_ylim(-0.01, 0.5)

    # Périodogramme Template
    template_in_range = (period_template >= period_min) & (
        period_template <= period_max
    )
    axes[1].plot(
        period_template[template_in_range],
        power_template[template_in_range],
        "orange",
        linewidth=1.5,
        label="Template",
    )
    axes[1].axvline(
        signal_period,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Signal injecté ({signal_period}j)",
    )
    axes[1].set_title("Périodogramme - Méthode Template")
    axes[1].set_ylabel("Puissance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(period_min, period_max)
    axes[1].set_ylim(-0.01, 0.5)

    # Périodogramme Masked Template
    masked_template_in_range = (period_masked_template >= period_min) & (
        period_masked_template <= period_max
    )
    axes[2].plot(
        period_masked_template[masked_template_in_range],
        power_masked_template[masked_template_in_range],
        "green",
        linewidth=1.5,
        label="Masked Template",
    )
    axes[2].axvline(
        signal_period,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Signal injecté ({signal_period}j)",
    )
    axes[2].set_title("Périodogramme - Méthode Masked Template")
    axes[2].set_xlabel("Période (jours)")
    axes[2].set_ylabel("Puissance")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(period_min, period_max)
    axes[2].set_ylim(-0.01, 0.5)
    plt.suptitle(
        f"Exp3: Périodogrammes complets (Signal: Kp={signal_amplitude} m/s, P={signal_period}j)",
        fontsize=14,
    )
    plt.tight_layout()

    # Créer le répertoire s'il n'existe pas
    import os

    os.makedirs("results/ccf/exp3", exist_ok=True)

    plt.savefig(
        "results/ccf/exp3/periodograms_full_range.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # === GRAPHIQUE 2: Zoom autour du signal ===
    zoom_window = 20  # ±20 jours autour du signal
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Zoom pour chaque méthode
    mask_zoom = (period_mask >= signal_period - zoom_window) & (
        period_mask <= signal_period + zoom_window
    )
    template_zoom = (period_template >= signal_period - zoom_window) & (
        period_template <= signal_period + zoom_window
    )
    masked_template_zoom = (period_masked_template >= signal_period - zoom_window) & (
        period_masked_template <= signal_period + zoom_window
    )

    ax.plot(
        period_mask[mask_zoom],
        power_mask[mask_zoom],
        "b-",
        linewidth=2,
        label="Mask",
        alpha=0.8,
    )
    ax.plot(
        period_template[template_zoom],
        power_template[template_zoom],
        "orange",
        linewidth=2,
        label="Template",
        alpha=0.8,
    )
    ax.plot(
        period_masked_template[masked_template_zoom],
        power_masked_template[masked_template_zoom],
        "green",
        linewidth=2,
        label="Masked Template",
        alpha=0.8,
    )

    ax.axvline(
        signal_period,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Signal injecté ({signal_period}j)",
    )
    ax.set_xlabel("Période (jours)")
    ax.set_ylabel("Puissance")
    ax.set_title(f"Zoom autour du signal planétaire (±{zoom_window}j)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(signal_period - zoom_window, signal_period + zoom_window)

    plt.tight_layout()
    plt.savefig("results/ccf/exp3/periodograms_zoom.png", dpi=300, bbox_inches="tight")
    plt.show()

    # === ANALYSE QUANTITATIVE ===
    print("\n=== ANALYSE QUANTITATIVE ===")

    # Calculer les performances pour chaque méthode
    tolerance = 3  # jours

    perf_mask = get_peak_value(
        periodo_mask, period_tolerance=tolerance, expected_period_for_peak=signal_period
    )
    perf_template = get_peak_value(
        periodo_template,
        period_tolerance=tolerance,
        expected_period_for_peak=signal_period,
    )
    perf_masked_template = get_peak_value(
        periodo_masked_template,
        period_tolerance=tolerance,
        expected_period_for_peak=signal_period,
    )

    print(f"Signal injecté: Kp = {signal_amplitude} m/s, P = {signal_period} jours")
    print(f"Tolérance de recherche: ±{tolerance} jours")
    print()
    print("PERFORMANCES:")
    print(
        f"Mask           : Power = {perf_mask[0]:.4f}, Période = {perf_mask[1]:.2f}j, Is_main = {bool(perf_mask[2])}"
    )
    print(
        f"Template       : Power = {perf_template[0]:.4f}, Période = {perf_template[1]:.2f}j, Is_main = {bool(perf_template[2])}"
    )
    print(
        f"Masked Template: Power = {perf_masked_template[0]:.4f}, Période = {perf_masked_template[1]:.2f}j, Is_main = {bool(perf_masked_template[2])}"
    )

    # === GRAPHIQUE 3: Barres de performance ===
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    methods = ["Mask", "Template", "Masked Template"]
    powers = [perf_mask[0], perf_template[0], perf_masked_template[0]]
    periods = [perf_mask[1], perf_template[1], perf_masked_template[1]]
    is_main_flags = [perf_mask[2], perf_template[2], perf_masked_template[2]]

    # Graphique des puissances
    colors = ["blue", "orange", "green"]
    bars1 = ax1.bar(methods, powers, color=colors, alpha=0.7)
    ax1.set_ylabel("Puissance du pic")
    ax1.set_title("Puissance de détection du signal")
    ax1.grid(True, alpha=0.3)

    # Ajouter les valeurs sur les barres
    for i, (bar, power) in enumerate(zip(bars1, powers)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{power:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Graphique des périodes détectées
    bars2 = ax2.bar(methods, periods, color=colors, alpha=0.7)
    ax2.axhline(
        signal_period,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Période vraie ({signal_period}j)",
    )
    ax2.set_ylabel("Période détectée (jours)")
    ax2.set_title("Précision de la période détectée")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Ajouter les valeurs sur les barres avec indication is_main
    for i, (bar, period, is_main) in enumerate(zip(bars2, periods, is_main_flags)):
        height = bar.get_height()
        main_indicator = "★" if is_main else "☆"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{period:.1f}j\n{main_indicator}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.suptitle("Performances de détection par méthode", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        "results/ccf/exp3/performance_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("\n✓ Graphiques sauvegardés dans results/ccf/exp3/")
    print("  - periodograms_full_range.png: Vue d'ensemble (1-1100j)")
    print(f"  - periodograms_zoom.png: Zoom autour du signal (±{zoom_window}j)")
    print("  - performance_comparison.png: Comparaison quantitative")
    print("\nLégende: ★ = pic principal, ☆ = pic secondaire")

    return {
        "signal_params": {"amplitude": signal_amplitude, "period": signal_period},
        "performances": {
            "mask": perf_mask,
            "template": perf_template,
            "masked_template": perf_masked_template,
        },
        "periodograms": {
            "mask": (period_mask, power_mask),
            "template": (period_template, power_template),
            "masked_template": (period_masked_template, power_masked_template),
        },
    }


if __name__ == "__main__":
    # Pour exécuter différentes expériences, décommentez la ligne correspondante:

    # main()           # Expérience principale
    # exp1()           # Expérience 1: variation de l'amplitude
    # exp2()           # Expérience 2: matrice amplitude vs période
    # plot_exp2()      # Visualisation de l'expérience 2
    exp3()  # Expérience 3: périodogrammes complets (Kp=1, P=100)
