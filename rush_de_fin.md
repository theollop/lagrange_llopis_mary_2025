- [ ] CCF vs AESTRA
	- [ ] Recoder proprement la CCF + docu
		- [ ] Comparaison méthode des masques avec méthode template + interpolation, génère un dataset de N spectres avec uniquement signal planétaire puis ensuite avec activité et dans les deux cas on compare les deux méthodes de CCF avec le périodo (amplitude du pic) + plot v_in vs v_out&
	- [ ] Recoder proprement AESTRA + docu
	- [ ] Batterie de test pour comparer les deux (Tester sur plusieurs dataset)
		- [ ] Précision (Injecter différentes vitesses et voir jusqu'à quand le pic ressort)
		- [ ] Rapidité (Mesurer temps de calcul)
		- [ ] Résistance au bruit

> 	Idéalement il faudrait pousser jusqu'à calculer sur toute la bande du spectre (Il faut que le code d'AESTRA soit optimisé pour pouvoir faire tourner sur plusieurs GPU du cluster dès Août ! )


- [ ] Performance et capacité d'AESTRA
	-> Répondre si AESTRA est capable de sortir les pics planètes sur un jeu de données comme le RV datachallenge (Grid Search sur tout les hyperparametres / Tester différentes combinaisons sans modifier l'architecture du papier -> il faut définir les limites du réseau et expliquer pourquoi il n'est pas assez performant)

- [ ]  Algo EM
	-> Implémenter un algo EM comme décrit par David et explorer cette méthode plus "statistique" sur les différents dataset, conclure quant à son efficacité relativement à la CCF et AESTRA + approfondir si résultats positifs sinon démontrer pourquoi ça n'est pas conceptuellement possible.

- [ ] Carte Blanche
	-> Explorer d'autres algo ML - DL afin de trouver potentiellement des pistes à approfondir et des méthodes plus performantes (Continuer le travail sur AESTRAM)