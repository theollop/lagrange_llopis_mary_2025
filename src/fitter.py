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

    Fonctionnalités d'analyse:
    - Ajustement de profils avec différents modèles
    - Calcul du bissecteur pour détecter les asymétries
    - Analyse de courbure du bissecteur
    - Comparaison de modèles automatique

    Le bissecteur est un outil crucial pour détecter les asymétries dans les profils
    de raies spectrales, particulièrement utile pour:
    - Détecter l'activité stellaire (taches, granulation)
    - Identifier les effets de convection
    - Distinguer les signaux planétaires des artefacts stellaires
    - Contrôler la qualité des mesures de vitesse radiale

    Détecte automatiquement l'orientation de la CCF (pic vers le haut ou vers le bas).
    """

    def __init__(self, plot_on_error=True):
        """
        Args:
            plot_on_error: Si True, plot automatiquement la CCF en cas d'erreur de fit
        """
        self.available_models = {
            "gaussian": self._gaussian,
            "lorentzian": self._lorentzian,
            "voigt": self._voigt,
            "polynomial": self._polynomial,
        }
        self.plot_on_error = plot_on_error

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

    def plot_ccf_on_error(
        self, ccf, v_grid, v_window, ccf_window, model, orientation, error_msg
    ):
        """
        Plot la CCF problématique en cas d'erreur de fit pour diagnostic.

        Args:
            ccf: Valeurs de la CCF complète
            v_grid: Grille de vitesses complète
            v_window: Grille de vitesses de la fenêtre de fit
            ccf_window: Valeurs de la CCF dans la fenêtre de fit
            model: Modèle qui a échoué
            orientation: Orientation du fit
            error_msg: Message d'erreur
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot CCF complète
        ax1.plot(v_grid, ccf, "b-", alpha=0.7, label="CCF complète")
        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax1.set_xlabel("Vitesse (m/s)")
        ax1.set_ylabel("CCF")
        ax1.set_title("CCF Complète")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot fenêtre de fit
        ax2.plot(v_window, ccf_window, "r-", linewidth=2, label="Fenêtre de fit")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax2.set_xlabel("Vitesse (m/s)")
        ax2.set_ylabel("CCF")
        ax2.set_title(f"Fenêtre de Fit (Erreur: {model})")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Ajout d'informations sur l'erreur
        info_text = (
            f"Modèle: {model}\nOrientation: {orientation}\nErreur: {error_msg[:50]}..."
        )
        ax2.text(
            0.02,
            0.98,
            info_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.suptitle(
            f"CCF Problématique - Échec du fit {model}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

        print("=== DIAGNOSTIC CCF ===")
        print(f"Modèle: {model}")
        print(f"Orientation: {orientation}")
        print(f"Erreur: {error_msg}")
        print(
            f"Valeurs CCF fenêtre - Min: {ccf_window.min():.6f}, Max: {ccf_window.max():.6f}"
        )
        print(f"Plage vitesses: {v_window.min():.1f} à {v_window.max():.1f} m/s")
        print(f"Nombre de points: {len(ccf_window)}")
        print("======================")

    def _estimate_initial_params(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        model: str,
        window_size: float,
        orientation: str,
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
            Tuple (paramètres initiaux)
        """
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

        return params

    def _create_fit_window(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        window_size: float,
        orientation: str,
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
        if orientation == "up":
            extremum_idx = np.argmax(ccf)
        else:
            extremum_idx = np.argmin(ccf)
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
        orientation: str,
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

        # Stockage temporaire pour le diagnostic en cas d'erreur
        self._current_ccf = ccf
        self._current_v_grid = v_grid

        # Crée la fenêtre de fit
        ccf_window, v_window, window_slice = self._create_fit_window(
            ccf, v_grid, window_size, orientation
        )

        # Estime les paramètres initiaux
        initial_params = self._estimate_initial_params(
            ccf_window, v_window, model, window_size, orientation, poly_order
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
            # Ajoute des paramètres par défaut pour curve_fit si pas spécifiés
            default_kwargs = {"maxfev": 2000}  # Augmente le nombre max d'évaluations
            default_kwargs.update(kwargs)  # Les kwargs utilisateur écrasent les défauts

            # Effectue l'ajustement
            popt, pcov = curve_fit(
                fit_func, v_window, ccf_window, p0=initial_params, **default_kwargs
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
            error_type = type(e).__name__
            if "maxfev" in str(e).lower():
                error_msg = f"Nombre maximum d'évaluations de fonction atteint ({error_type}): {e}"
            else:
                error_msg = f"Erreur d'ajustement ({error_type}): {e}"

            results = {
                "model": model,
                "orientation": orientation,
                "window_size": window_size,
                "v_window": v_window,
                "ccf_window": ccf_window,
                "success": False,
                "error_message": error_msg,
                "r_squared": -np.inf,  # Valeur très faible pour le tri
            }

            # Plot automatique en cas d'erreur si activé
            if self.plot_on_error:
                # On a besoin de récupérer la CCF complète pour le plot
                # Pour cela, on va l'ajouter aux paramètres du fit
                self.plot_ccf_on_error(
                    ccf=getattr(
                        self, "_current_ccf", ccf_window
                    ),  # CCF complète si disponible
                    v_grid=getattr(
                        self, "_current_v_grid", v_window
                    ),  # V_grid complet si disponible
                    v_window=v_window,
                    ccf_window=ccf_window,
                    model=model,
                    orientation=orientation,
                    error_msg=error_msg,
                )

            warnings.warn(f"Échec de l'ajustement {model}: {error_msg}")

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
        orientation: str,
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
            try:
                result = self.fit_ccf(
                    ccf,
                    v_grid,
                    model=model,
                    window_size=window_size,
                    poly_order=poly_order,
                    orientation=orientation,
                )
                results[model] = result
            except Exception as e:
                print(f"Erreur lors de l'ajustement {model}: {e}")
                # Ajouter un résultat d'échec pour ce modèle
                results[model] = {
                    "model": model,
                    "orientation": orientation,
                    "window_size": window_size,
                    "success": False,
                    "error_message": str(e),
                    "r_squared": -np.inf,  # Valeur très faible pour ne pas être sélectionné
                }

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

    def find_best_fit(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        orientation: str,
        window_size: float = 2000,
        poly_order: int = 2,
    ) -> Dict:
        """
        Trouve le meilleur modèle d'ajustement pour la CCF.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses
            window_size: Taille de la fenêtre de fit
            poly_order: Ordre du polynôme

        Returns:
            Dictionnaire contenant les résultats du meilleur ajustement
        """
        comparison_results = self.compare_models(
            ccf,
            v_grid,
            window_size=window_size,
            poly_order=poly_order,
            orientation=orientation,
        )

        # Filtre les résultats réussis seulement
        successful_results = {
            model: result
            for model, result in comparison_results.items()
            if result.get("success", False)
        }

        if not successful_results:
            # Aucun ajustement n'a réussi, retourner le résultat d'échec le moins mauvais
            print(
                "Aucun ajustement n'a réussi. Retour du résultat avec la meilleure r_squared disponible."
            )
            best_model = max(
                comparison_results.items(), key=lambda x: x[1].get("r_squared", -np.inf)
            )
            return best_model[1]

        # Sélectionne le modèle avec le meilleur R² parmi les succès
        best_model = max(successful_results.items(), key=lambda x: x[1]["r_squared"])

        return best_model[1]

    def fit_ccf_with_best_fit(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        orientation: str,
        window_size: float = 2000,
        poly_order: int = 2,
    ) -> Dict:
        """
        Ajuste la CCF en utilisant le meilleur modèle trouvé.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses
            window_size: Taille de la fenêtre de fit
            poly_order: Ordre du polynôme

        Returns:
            Dictionnaire contenant les résultats de l'ajustement
        """
        best_fit = self.find_best_fit(
            ccf,
            v_grid,
            window_size=window_size,
            poly_order=poly_order,
            orientation=orientation,
        )
        return best_fit

    def compute_bisector(
        self,
        ccf: np.ndarray,
        v_grid: np.ndarray,
        window_size: float = 2000,
        orientation: str = "down",
        depth_levels: np.ndarray = None,
        interpolation_method: str = "linear",
    ) -> Dict:
        """
        Calcule le bissecteur de la CCF.

        Le bissecteur est calculé en trouvant les points médians entre les deux côtés
        de la CCF à différents niveaux de profondeur.

        Args:
            ccf: Valeurs de la CCF
            v_grid: Grille de vitesses en m/s
            window_size: Taille de la fenêtre de calcul en m/s
            orientation: Orientation de la CCF ("up" pour pic vers le haut, "down" pour pic vers le bas)
            depth_levels: Niveaux de profondeur relatifs pour le calcul (0 à 1)
                         Si None, utilise des niveaux par défaut
            interpolation_method: Méthode d'interpolation ("linear", "cubic")

        Returns:
            Dictionnaire contenant:
                - bisector_v: Vitesses du bissecteur
                - bisector_depth: Profondeurs correspondantes
                - depth_levels: Niveaux utilisés
                - ccf_window: CCF dans la fenêtre
                - v_window: Grille de vitesses de la fenêtre
                - center_velocity: Vitesse centrale du profil
                - success: Succès du calcul
        """
        try:
            # Crée la fenêtre de calcul
            ccf_window, v_window, _ = self._create_fit_window(
                ccf, v_grid, window_size, orientation
            )

            # Définit les niveaux de profondeur par défaut
            if depth_levels is None:
                depth_levels = np.linspace(0.1, 0.9, 17)  # 17 niveaux de 10% à 90%

            # Trouve le minimum/maximum et le niveau de base
            if orientation == "down":
                extremum_idx = np.argmin(ccf_window)
                extremum_value = ccf_window[extremum_idx]
                baseline = 1
                # Pour un pic vers le bas, les niveaux vont du baseline vers l'extremum
                depth_values = baseline + depth_levels * (extremum_value - baseline)
            else:
                extremum_idx = np.argmax(ccf_window)
                extremum_value = ccf_window[extremum_idx]
                baseline = 0
                # Pour un pic vers le haut, les niveaux vont du baseline vers l'extremum
                depth_values = baseline + depth_levels * (extremum_value - baseline)

            center_velocity = v_window[extremum_idx]

            # Calcule le bissecteur
            bisector_v = []
            bisector_depth = []

            for depth_value in depth_values:
                # Trouve les intersections avec ce niveau
                intersections = self._find_intersections(
                    v_window, ccf_window, depth_value
                )

                if len(intersections) >= 2:
                    # Trouve les deux intersections les plus proches du centre
                    # Une de chaque côté
                    left_intersections = [
                        v for v in intersections if v < center_velocity
                    ]
                    right_intersections = [
                        v for v in intersections if v > center_velocity
                    ]

                    if left_intersections and right_intersections:
                        # Prend les intersections les plus proches du centre
                        left_v = max(left_intersections)
                        right_v = min(right_intersections)

                        # Point médian du bissecteur
                        bisector_point = (left_v + right_v) / 2
                        bisector_v.append(bisector_point)
                        bisector_depth.append(depth_value)

            if len(bisector_v) == 0:
                raise ValueError(
                    "Impossible de calculer le bissecteur - pas assez d'intersections trouvées"
                )

            bisector_v = np.array(bisector_v)
            bisector_depth = np.array(bisector_depth)

            # Calcule quelques statistiques du bissecteur
            bisector_span = np.max(bisector_v) - np.min(bisector_v)
            bisector_slope = self._compute_bisector_slope(bisector_v, bisector_depth)

            results = {
                "bisector_v": bisector_v,
                "bisector_depth": bisector_depth,
                "depth_levels": depth_levels,
                "ccf_window": ccf_window,
                "v_window": v_window,
                "center_velocity": center_velocity,
                "bisector_span": bisector_span,
                "bisector_slope": bisector_slope,
                "window_size": window_size,
                "orientation": orientation,
                "success": True,
            }

            return results

        except Exception as e:
            return {
                "success": False,
                "error_message": f"Erreur lors du calcul du bissecteur: {e}",
                "window_size": window_size,
                "orientation": orientation,
            }

    def _find_intersections(
        self, v_grid: np.ndarray, ccf: np.ndarray, level: float
    ) -> list:
        """
        Trouve les intersections entre la CCF et un niveau horizontal donné.

        Args:
            v_grid: Grille de vitesses
            ccf: Valeurs de la CCF
            level: Niveau horizontal

        Returns:
            Liste des vitesses d'intersection
        """
        intersections = []

        # Trouve les changements de signe de (ccf - level)
        diff = ccf - level
        sign_changes = np.where(np.diff(np.signbit(diff)))[0]

        for i in sign_changes:
            # Interpolation linéaire pour trouver l'intersection précise
            if i + 1 < len(ccf):
                v1, v2 = v_grid[i], v_grid[i + 1]
                ccf1, ccf2 = ccf[i], ccf[i + 1]

                # Intersection par interpolation linéaire
                if ccf2 != ccf1:  # Évite la division par zéro
                    v_intersect = v1 + (level - ccf1) * (v2 - v1) / (ccf2 - ccf1)
                    intersections.append(v_intersect)

        return intersections

    def _compute_bisector_slope(
        self, bisector_v: np.ndarray, bisector_depth: np.ndarray
    ) -> float:
        """
        Calcule la pente moyenne du bissecteur.

        Args:
            bisector_v: Vitesses du bissecteur
            bisector_depth: Profondeurs du bissecteur

        Returns:
            Pente en m/s par unité de profondeur
        """
        if len(bisector_v) < 2:
            return np.nan

        # Ajustement linéaire simple
        try:
            from scipy.stats import linregress

            slope, _, _, _, _ = linregress(bisector_depth, bisector_v)
            return slope
        except Exception:
            # Fallback sur une pente simple
            return (bisector_v[-1] - bisector_v[0]) / (
                bisector_depth[-1] - bisector_depth[0]
            )

    def plot_bisector(
        self,
        bisector_result: Dict,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        show_levels: bool = True,
    ):
        """
        Trace la CCF avec son bissecteur.

        Args:
            bisector_result: Résultat de compute_bisector
            title: Titre optionnel
            figsize: Taille de la figure
            show_levels: Afficher les niveaux de profondeur utilisés
        """
        if not bisector_result["success"]:
            print(
                f"Impossible de tracer - calcul du bissecteur échoué: {bisector_result['error_message']}"
            )
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Graphique principal: CCF + bissecteur
        ax1.plot(
            bisector_result["v_window"],
            bisector_result["ccf_window"],
            "b-",
            linewidth=2,
            label="CCF",
        )

        # Trace le bissecteur
        ax1.plot(
            bisector_result["bisector_v"],
            bisector_result["bisector_depth"],
            "ro-",
            linewidth=2,
            markersize=4,
            label="Bissecteur",
        )

        # Marque le centre
        ax1.axvline(
            bisector_result["center_velocity"],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Centre",
        )

        # Affiche les niveaux de profondeur si demandé
        if show_levels:
            for level in bisector_result["bisector_depth"]:
                ax1.axhline(level, color="gray", alpha=0.3, linestyle=":")

        ax1.set_xlabel("Vitesse radiale (m/s)")
        ax1.set_ylabel("CCF / Profondeur")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if title is None:
            title = "CCF et Bissecteur"
        ax1.set_title(title)

        # Graphique du bissecteur seul
        ax2.plot(
            bisector_result["bisector_v"],
            bisector_result["bisector_depth"],
            "ro-",
            linewidth=2,
            markersize=4,
        )

        ax2.axvline(
            bisector_result["center_velocity"],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Centre",
        )

        ax2.set_xlabel("Vitesse du bissecteur (m/s)")
        ax2.set_ylabel("Profondeur")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Bissecteur")

        # Ajoute les statistiques
        stats_text = (
            f"Span: {bisector_result['bisector_span']:.2f} m/s\n"
            f"Pente: {bisector_result['bisector_slope']:.2e} m/s/unité"
        )
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def analyze_bisector_curvature(
        self,
        bisector_result: Dict,
        poly_order: int = 2,
    ) -> Dict:
        """
        Analyse la courbure du bissecteur en ajustant un polynôme.

        Args:
            bisector_result: Résultat de compute_bisectoré
            poly_order: Ordre du polynôme à ajuster

        Returns:
            Dictionnaire avec l'analyse de courbure
        """
        if not bisector_result["success"]:
            return {"success": False, "error_message": "Bissecteur invalide"}

        try:
            bisector_v = bisector_result["bisector_v"]
            bisector_depth = bisector_result["bisector_depth"]

            if len(bisector_v) < poly_order + 1:
                return {
                    "success": False,
                    "error_message": f"Pas assez de points pour un polynôme d'ordre {poly_order}",
                }

            # Ajustement polynomial
            coeffs = np.polyfit(bisector_depth, bisector_v, poly_order)
            poly_fit = np.polyval(coeffs, bisector_depth)

            # Calcule les résidus
            residuals = bisector_v - poly_fit
            rms_residuals = np.sqrt(np.mean(residuals**2))

            # Calcule la courbure (dérivée seconde) si ordre >= 2
            curvature = np.nan
            if poly_order >= 2:
                # Dérivée seconde est le coefficient de x² multiplié par 2
                curvature = 2 * coeffs[-3]  # Coefficient de x²

            return {
                "success": True,
                "poly_coeffs": coeffs,
                "poly_fit": poly_fit,
                "residuals": residuals,
                "rms_residuals": rms_residuals,
                "curvature": curvature,
                "poly_order": poly_order,
            }

        except Exception as e:
            return {
                "success": False,
                "error_message": f"Erreur lors de l'analyse de courbure: {e}",
            }


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


def quick_bisector_analysis(
    ccf: np.ndarray,
    v_grid: np.ndarray,
    window_size: float = 2000,
    orientation: str = "down",
) -> Dict:
    """
    Fonction de commodité pour une analyse rapide du bissecteur.

    Args:
        ccf: Valeurs de la CCF
        v_grid: Grille de vitesses
        window_size: Taille de la fenêtre de calcul
        orientation: Orientation de la CCF

    Returns:
        Dictionnaire avec les résultats du bissecteur
    """
    fitter = CCFFitter()
    return fitter.compute_bisector(
        ccf, v_grid, window_size=window_size, orientation=orientation
    )


def extract_bisector_metrics(bisector_result: Dict) -> Dict:
    """
    Extrait les métriques principales du bissecteur.

    Args:
        bisector_result: Résultat de compute_bisector

    Returns:
        Dictionnaire avec les métriques principales:
        - bisector_span: Étendue du bissecteur en m/s
        - bisector_slope: Pente du bissecteur
        - bisector_rms: RMS de la dispersion du bissecteur
        - bisector_mean_velocity: Vitesse moyenne du bissecteur
    """
    if not bisector_result["success"]:
        return {
            "success": False,
            "error_message": bisector_result.get(
                "error_message", "Bissecteur invalide"
            ),
        }

    bisector_v = bisector_result["bisector_v"]
    center_velocity = bisector_result["center_velocity"]

    # Calcule les métriques
    bisector_span = bisector_result["bisector_span"]
    bisector_slope = bisector_result["bisector_slope"]
    bisector_mean_velocity = np.mean(bisector_v)
    bisector_rms = np.sqrt(np.mean((bisector_v - bisector_mean_velocity) ** 2))

    # Décalage moyen par rapport au centre
    bisector_offset = bisector_mean_velocity - center_velocity

    return {
        "success": True,
        "bisector_span": bisector_span,
        "bisector_slope": bisector_slope,
        "bisector_rms": bisector_rms,
        "bisector_mean_velocity": bisector_mean_velocity,
        "bisector_offset": bisector_offset,
        "center_velocity": center_velocity,
        "n_points": len(bisector_v),
    }


def get_sin_error(rvs, true_period, true_amplitude):
    # 1) retirer l'offset
    x_c = rvs - np.mean(rvs)

    # 2) FFT et pic spectral
    X = np.fft.fft(x_c)
    k = np.argmax(np.abs(X[: len(rvs) // 2]))  # moitié positive
    f_est = k * 1 / len(rvs)
    A_est = 2 * np.abs(X[k]) / len(rvs)

    # 3) calculer les erreurs
    delta_f = f_est - 1 / true_period
    delta_A = A_est - true_amplitude

    # 4) Erreurs en pourcentage
    if true_period != 0:
        delta_f = np.abs(delta_f / (1 / true_period) * 100)
    if true_amplitude != 0:
        delta_A = np.abs(delta_A / true_amplitude * 100)

    # 4) retourner les erreurs
    return delta_f, delta_A


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
        CCFs_template[0], v_grid, model="gaussian", window_size=2000, orientation="down"
    )
    if gauss_result["success"]:
        rv, rv_err = extract_rv_from_fit(gauss_result)
        print(f"   VR = {rv:.2f} ± {rv_err:.2f} m/s")
        print(f"   R² = {gauss_result['r_squared']:.4f}")

    # Test comparaison de modèles
    print("\n2. Comparaison de modèles:")
    comparison = fitter.compare_models(
        CCFs_template[0], v_grid, window_size=2000, poly_order=14, orientation="down"
    )

    # Test calcul du bissecteur
    print("\n3. Calcul du bissecteur:")
    bisector_result = fitter.compute_bisector(
        CCFs_template[0], v_grid, window_size=2000, orientation="down"
    )
    if bisector_result["success"]:
        print(f"   Span du bissecteur: {bisector_result['bisector_span']:.2f} m/s")
        print(
            f"   Pente du bissecteur: {bisector_result['bisector_slope']:.2e} m/s/unité"
        )
        print(f"   Nombre de points: {len(bisector_result['bisector_v'])}")

        # Analyse de courbure
        curvature_analysis = fitter.analyze_bisector_curvature(
            bisector_result, poly_order=2
        )
        if curvature_analysis["success"]:
            print(f"   Courbure: {curvature_analysis['curvature']:.2e}")
            print(f"   RMS résidus: {curvature_analysis['rms_residuals']:.2e} m/s")

    # Trace les résultats si matplotlib est disponible
    try:
        fitter.plot_fit_result(gauss_result, title="Test ajustement gaussien")
        fitter.plot_model_comparison(comparison)

        # Trace le bissecteur si le calcul a réussi
        if bisector_result["success"]:
            fitter.plot_bisector(bisector_result, title="Analyse du bissecteur CCF")
    except Exception as e:
        print(f"Impossible d'afficher les graphiques: {e}")

    print("\nTest terminé!")
