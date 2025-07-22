import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
from typing import Tuple, Dict, Optional
import warnings


class CCFFitter:
    """
    Classe pour ajuster différents profils sur une fonction de corrélation croisée (CCF).

    Supporte les profils suivants:
    - Gaussienne
    - Lorentzienne
    - Voigt
    - Polynomial (ordre variable)

    Détecte automatiquement l'orientation de la CCF (pic vers le haut ou vers le bas).
    """

    def __init__(self):
        self.available_models = {
            "gaussian": self._gaussian,
            "lorentzian": self._lorentzian,
            "voigt": self._voigt,
            "polynomial": self._polynomial,
        }

    @staticmethod
    def _gaussian(v, amplitude, center, sigma, offset=0):
        """Profil gaussien."""
        return amplitude * np.exp(-0.5 * ((v - center) / sigma) ** 2) + offset

    @staticmethod
    def _lorentzian(v, amplitude, center, gamma, offset=0):
        """Profil lorentzien."""
        return amplitude * gamma**2 / ((v - center) ** 2 + gamma**2) + offset

    @staticmethod
    def _voigt(v, amplitude, center, sigma, gamma, offset=0):
        """
        Profil de Voigt (convolution gaussienne × lorentzienne).
        Utilise la fonction de Faddeeva pour le calcul.
        """
        z = ((v - center) + 1j * gamma) / (sigma * np.sqrt(2))
        w = wofz(z)
        return amplitude * np.real(w) / (sigma * np.sqrt(2 * np.pi)) + offset

    @staticmethod
    def _polynomial(v, *coeffs):
        """Profil polynomial d'ordre variable."""
        return np.polyval(coeffs, v)

    def _detect_ccf_orientation(self, ccf: np.ndarray, v_grid: np.ndarray) -> str:
        """
        Détecte si la CCF a un pic vers le haut (absorption) ou vers le bas (émission).

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses correspondante

        Returns:
            'up' si pic vers le haut, 'down' si pic vers le bas
        """
        # Trouve l'extremum principal
        max_idx = np.argmax(np.abs(ccf))
        max_val = ccf[max_idx]

        # Calcule la médiane pour estimer le niveau de base
        median_val = np.median(ccf)

        # Si la valeur maximale absolue est au-dessus de la médiane -> pic vers le haut
        # Sinon -> pic vers le bas
        if max_val > median_val:
            return "up"
        else:
            return "down"

    def _estimate_initial_params(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        model: str,
        window_size: float,
        poly_order: int = 2,
    ) -> Tuple[list, str]:
        """
        Estime les paramètres initiaux pour l'ajustement selon le modèle choisi.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses
            model: Type de modèle ('gaussian', 'lorentzian', 'voigt', 'polynomial')
            window_size: Taille de la fenêtre de fit
            poly_order: Ordre du polynôme (pour model='polynomial')

        Returns:
            Tuple (paramètres initiaux, orientation)
        """
        orientation = self._detect_ccf_orientation(ccf, v_grid)

        # Trouve l'extremum et sa position
        if orientation == "up":
            extremum_idx = np.argmax(ccf)
        else:
            extremum_idx = np.argmin(ccf)

        center_guess = v_grid[extremum_idx]
        amplitude_guess = ccf[extremum_idx]
        offset_guess = np.median(ccf)

        # Estime la largeur du pic en trouvant la largeur à mi-hauteur
        half_max = offset_guess + (amplitude_guess - offset_guess) / 2
        if orientation == "down":
            half_max = offset_guess + (amplitude_guess - offset_guess) / 2

        # Trouve les indices où ccf traverse half_max
        if orientation == "up":
            indices = np.where(ccf > half_max)[0]
        else:
            indices = np.where(ccf < half_max)[0]

        if len(indices) > 1:
            width_guess = v_grid[indices[-1]] - v_grid[indices[0]]
        else:
            width_guess = window_size / 4  # Estimation par défaut

        sigma_guess = width_guess / (2 * np.sqrt(2 * np.log(2)))  # FWHM -> sigma
        gamma_guess = width_guess / 2  # Pour Lorentzienne et Voigt

        if model == "gaussian":
            params = [amplitude_guess, center_guess, sigma_guess, offset_guess]
        elif model == "lorentzian":
            params = [amplitude_guess, center_guess, gamma_guess, offset_guess]
        elif model == "voigt":
            params = [
                amplitude_guess,
                center_guess,
                sigma_guess,
                gamma_guess,
                offset_guess,
            ]
        elif model == "polynomial":
            # Pour le polynôme, on estime grossièrement
            params = [amplitude_guess] + [0] * poly_order

        return params, orientation

    def _create_fit_window(
        self, ccf: np.ndarray, v_grid: np.ndarray, window_size: float
    ) -> Tuple[np.ndarray, np.ndarray, slice]:
        """
        Crée une fenêtre centrée sur le pic principal pour l'ajustement.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses
            window_size: Taille de la fenêtre en m/s

        Returns:
            Tuple (ccf_window, v_window, slice_indices)
        """
        # Trouve le centre du pic
        extremum_idx = np.argmax(np.abs(ccf))
        center_v = v_grid[extremum_idx]

        # Définit la fenêtre
        v_min = center_v - window_size / 2
        v_max = center_v + window_size / 2

        # Trouve les indices correspondants
        mask = (v_grid >= v_min) & (v_grid <= v_max)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(
                f"Aucun point dans la fenêtre [{v_min:.0f}, {v_max:.0f}] m/s"
            )

        slice_obj = slice(indices[0], indices[-1] + 1)
        return ccf[mask], v_grid[mask], slice_obj

    def fit_ccf(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        model: str = "gaussian",
        window_size: float = 2000,
        poly_order: int = 2,
        **kwargs,
    ) -> Dict:
        """
        Ajuste un profil sur la CCF dans une fenêtre donnée.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses en m/s
            model: Type de modèle ('gaussian', 'lorentzian', 'voigt', 'polynomial')
            window_size: Taille de la fenêtre de fit en m/s
            poly_order: Ordre du polynôme (pour model='polynomial')
            **kwargs: Arguments supplémentaires pour curve_fit

        Returns:
            Dictionnaire contenant les résultats de l'ajustement
        """
        if model not in self.available_models:
            raise ValueError(
                f"Modèle '{model}' non supporté. Modèles disponibles: {list(self.available_models.keys())}"
            )

        # Crée la fenêtre de fit
        ccf_window, v_window, window_slice = self._create_fit_window(
            ccf, v_grid, window_size
        )

        # Estime les paramètres initiaux
        initial_params, orientation = self._estimate_initial_params(
            ccf_window, v_window, model, window_size, poly_order
        )

        # Prépare la fonction de fit
        if model == "polynomial":
            # Pour le polynôme, on utilise une fonction wrapper avec l'ordre correct
            def fit_func(x, *coeffs):
                return self._polynomial(x, *coeffs)

            # Ajuste le nombre de paramètres initiaux selon l'ordre
            initial_params = [0] * (poly_order + 1)
            initial_params[0] = np.mean(ccf_window)  # Terme constant
        else:
            fit_func = self.available_models[model]

        try:
            # Effectue l'ajustement
            popt, pcov = curve_fit(
                fit_func, v_window, ccf_window, p0=initial_params, **kwargs
            )

            # Calcule le modèle ajusté
            ccf_fit = fit_func(v_window, *popt)

            # Calcule les statistiques
            residuals = ccf_window - ccf_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ccf_window - np.mean(ccf_window)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rms = np.sqrt(np.mean(residuals**2))

            # Erreurs sur les paramètres
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else None

            # Organise les résultats selon le modèle
            results = {
                "model": model,
                "orientation": orientation,
                "window_size": window_size,
                "v_window": v_window,
                "ccf_window": ccf_window,
                "ccf_fit": ccf_fit,
                "residuals": residuals,
                "parameters": popt,
                "parameter_errors": param_errors,
                "covariance": pcov,
                "r_squared": r_squared,
                "rms": rms,
                "success": True,
            }

            # Ajoute les noms des paramètres selon le modèle
            if model == "gaussian":
                param_names = ["amplitude", "center", "sigma", "offset"]
            elif model == "lorentzian":
                param_names = ["amplitude", "center", "gamma", "offset"]
            elif model == "voigt":
                param_names = ["amplitude", "center", "sigma", "gamma", "offset"]
            elif model == "polynomial":
                param_names = [f"coeff_{i}" for i in range(len(popt))]

            results["parameter_names"] = param_names

            # Crée un dictionnaire paramètre -> valeur
            results["fitted_params"] = dict(zip(param_names, popt))
            if param_errors is not None:
                results["param_errors_dict"] = dict(zip(param_names, param_errors))

        except Exception as e:
            # En cas d'échec de l'ajustement
            results = {
                "model": model,
                "orientation": orientation,
                "window_size": window_size,
                "v_window": v_window,
                "ccf_window": ccf_window,
                "success": False,
                "error_message": str(e),
            }
            warnings.warn(f"Échec de l'ajustement {model}: {e}")

        return results

    def plot_fit_result(
        self,
        results: Dict,
        title: Optional[str] = None,
        show_residuals: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Trace les résultats de l'ajustement.

        Args:
            results: Dictionnaire de résultats retourné par fit_ccf
            title: Titre optionnel pour le graphique
            show_residuals: Afficher les résidus
            figsize: Taille de la figure
        """
        if not results["success"]:
            print(
                f"Impossible de tracer - ajustement échoué: {results['error_message']}"
            )
            return

        if show_residuals:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Graphique principal
        ax1.plot(
            results["v_window"],
            results["ccf_window"],
            "bo-",
            label="CCF données",
            markersize=3,
        )
        ax1.plot(
            results["v_window"],
            results["ccf_fit"],
            "r-",
            label=f"Ajustement {results['model']}",
            linewidth=2,
        )

        ax1.set_xlabel("Vitesse radiale (m/s)")
        ax1.set_ylabel("CCF")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Titre avec statistiques
        if title is None:
            title = f"Ajustement CCF - {results['model'].capitalize()}"
        title += f" (R² = {results['r_squared']:.4f}, RMS = {results['rms']:.2e})"
        ax1.set_title(title)

        # Affiche les paramètres ajustés
        param_text = []
        for name, value in results["fitted_params"].items():
            if "param_errors_dict" in results:
                error = results["param_errors_dict"][name]
                param_text.append(f"{name}: {value:.2f} ± {error:.2f}")
            else:
                param_text.append(f"{name}: {value:.2f}")

        ax1.text(
            0.02,
            0.98,
            "\n".join(param_text),
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Graphique des résidus
        if show_residuals:
            ax2.plot(results["v_window"], results["residuals"], "go-", markersize=3)
            ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Vitesse radiale (m/s)")
            ax2.set_ylabel("Résidus")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_models(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        models: list = None,
        window_size: float = 2000,
        poly_order: int = 2,
    ) -> Dict:
        """
        Compare plusieurs modèles d'ajustement sur la même CCF.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses
            models: Liste des modèles à comparer (par défaut tous sauf polynomial)
            window_size: Taille de la fenêtre de fit
            poly_order: Ordre du polynôme

        Returns:
            Dictionnaire contenant les résultats pour chaque modèle
        """
        if models is None:
            models = ["gaussian", "lorentzian", "voigt", "polynomial"]

        results = {}
        for model in models:
            print(f"Ajustement {model}...")
            try:
                result = self.fit_ccf(
                    ccf,
                    v_grid,
                    model=model,
                    window_size=window_size,
                    poly_order=poly_order,
                )
                results[model] = result
            except Exception as e:
                print(f"Erreur lors de l'ajustement {model}: {e}")

        return results

    def plot_model_comparison(
        self,
        comparison_results: Dict,
        title: str = "Comparaison des modèles d'ajustement",
    ):
        """
        Trace une comparaison des différents modèles ajustés.

        Args:
            comparison_results: Résultats de compare_models
            title: Titre du graphique
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        colors = ["blue", "red", "green", "orange", "purple"]

        # Trace les données originales une seule fois
        first_result = next(iter(comparison_results.values()))
        if first_result["success"]:
            ax.plot(
                first_result["v_window"],
                first_result["ccf_window"],
                "ko-",
                label="CCF données",
                markersize=3,
                alpha=0.7,
            )

        # Trace chaque ajustement
        for i, (model, result) in enumerate(comparison_results.items()):
            if result["success"]:
                color = colors[i % len(colors)]
                ax.plot(
                    result["v_window"],
                    result["ccf_fit"],
                    "-",
                    color=color,
                    linewidth=2,
                    label=f"{model} (R²={result['r_squared']:.4f})",
                )

        ax.set_xlabel("Vitesse radiale (m/s)")
        ax.set_ylabel("CCF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        # Affiche un résumé statistique
        print("\n" + "=" * 60)
        print("RÉSUMÉ DE LA COMPARAISON")
        print("=" * 60)
        for model, result in comparison_results.items():
            if result["success"]:
                print(
                    f"{model.upper():>12}: R² = {result['r_squared']:.6f}, RMS = {result['rms']:.2e}"
                )
            else:
                print(f"{model.upper():>12}: ÉCHEC")


# Fonctions utilitaires
def quick_gaussian_fit(
    ccf: np.ndarray, v_grid: np.ndarray, window_size: float = 2000
) -> Dict:
    """Fonction de commodité pour un ajustement gaussien rapide."""
    fitter = CCFFitter()
    return fitter.fit_ccf(ccf, v_grid, model="gaussian", window_size=window_size)


def quick_voigt_fit(
    ccf: np.ndarray, v_grid: np.ndarray, window_size: float = 2000
) -> Dict:
    """Fonction de commodité pour un ajustement Voigt rapide."""
    fitter = CCFFitter()
    return fitter.fit_ccf(ccf, v_grid, model="voigt", window_size=window_size)


def extract_rv_from_fit(fit_result: Dict) -> Tuple[float, float]:
    """
    Extrait la vitesse radiale et son erreur d'un résultat d'ajustement.

    Args:
        fit_result: Résultat de fit_ccf

    Returns:
        Tuple (vitesse_radiale, erreur_vitesse)
    """
    if not fit_result["success"]:
        raise ValueError("Impossible d'extraire la VR - ajustement échoué")

    model = fit_result["model"]
    params = fit_result["fitted_params"]

    if model in ["gaussian", "lorentzian", "voigt"]:
        rv = params["center"]
        if "param_errors_dict" in fit_result:
            rv_error = fit_result["param_errors_dict"]["center"]
        else:
            rv_error = np.nan
    elif model == "polynomial":
        # Pour un polynôme, on trouve le minimum/maximum
        coeffs = fit_result["parameters"]
        # Dérivée du polynôme
        deriv_coeffs = np.polyder(coeffs)
        roots = np.roots(deriv_coeffs)
        # Prend la racine réelle la plus proche du centre de la fenêtre
        real_roots = roots[np.isreal(roots)].real
        if len(real_roots) > 0:
            center_window = np.mean(fit_result["v_window"])
            rv = real_roots[np.argmin(np.abs(real_roots - center_window))]
            rv_error = np.nan  # Difficile à estimer pour un polynôme
        else:
            rv = np.nan
            rv_error = np.nan
    else:
        rv = np.nan
        rv_error = np.nan

    return rv, rv_error


# Exemple d'utilisation
if __name__ == "__main__":
    # Test avec des données simulées
    np.random.seed(42)

    # Crée une CCF simulée (gaussienne + bruit)
    from io_spec import get_specs_from_h5
    from ccf import compute_CCF_mask, compute_ccf_template
    import torch

    # Chargement des données
    specs, wavegrid = get_specs_from_h5(
        filepath="/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/spec_cube_tot.h5",
        idx_spec_start=0,
        idx_spec_end=5,  # Seulement quelques spectres pour l'exemple
        wavemin=None,
        wavemax=None,
    )

    template = specs[0]
    specs_to_analyze = specs[1:]
    v_grid = np.arange(-20000, 20000, 100)  # Grille de vitesses

    CCFs_template = compute_ccf_template(
        specs=specs_to_analyze,  # Un seul spectre
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        dtype=torch.float32,
    )

    CCFs_mask = compute_CCF_mask(
        specs=specs_to_analyze,
        wavegrid=wavegrid,
        v_grid=v_grid,
        window_size_velocity=820,
    )

    print(CCFs_template)

    print("Test du fitter CCF")
    print("=" * 50)

    # Crée le fitter
    fitter = CCFFitter()

    # Test ajustement gaussien
    print("1. Ajustement gaussien:")
    gauss_result = fitter.fit_ccf(
        CCFs_template[0], v_grid, model="gaussian", window_size=2000
    )
    if gauss_result["success"]:
        rv, rv_err = extract_rv_from_fit(gauss_result)
        print(f"   VR = {rv:.2f} ± {rv_err:.2f} m/s")
        print(f"   R² = {gauss_result['r_squared']:.4f}")

    # Test comparaison de modèles
    print("\n2. Comparaison de modèles:")
    comparison = fitter.compare_models(
        CCFs_template[0], v_grid, window_size=2000, poly_order=14
    )

    # Trace les résultats si matplotlib est disponible
    try:
        fitter.plot_fit_result(gauss_result, title="Test ajustement gaussien")
        fitter.plot_model_comparison(comparison)
    except Exception as e:
        print(f"Impossible d'afficher les graphiques: {e}")

    print("\nTest terminé!")
