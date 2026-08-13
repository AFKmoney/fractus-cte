# Fractus — l'histoire derrière l'architecture

*Récit en première personne. Signé Philippe-Antoine Robert.*

---

## Le moment où j'ai fermé les yeux

Tout a commencé par un geste simple. J'ai fermé les yeux, et au lieu de chercher un papier à lire, j'ai regardé ce qui se passait derrière mes paupières. Mes pensées.

Je ne les ai pas analysées comme un psychologue les analyse. Je les ai regardées comme un physicien regarde un phénomène — avec la demande naïve et totale de les comprendre mécaniquement. Et ce que j'ai vu, c'est que **mes pensées ne fonctionnent pas du tout comme un transformer.**

Un transformer, c'est : on lui donne une entrée, il fait un seul grand calcul, il crache une sortie. Entrée → fonction → sortie. C'est une machine à réponse, pas une machine à penser.

Mais quand j'ai observé mes propres pensées, voici ce que j'ai vu :

**Mes pensées sont continues.** Elles ne s'arrêtent jamais. Même quand je me tais, même dans le silence, il y a quelque chose qui tourne — un fond, un babil, une activité qui précède la parole. Je ne « calcule » pas une réponse : je la laisse émerger d'un courant qui coulait déjà avant la question.

**Mes pensées s'accumulent.** Chaque chose que je perçois ne repart pas de zéro. Elle s'ajoute à un état qui était déjà là, qui contient tout ce que j'ai vécu dans la seconde, la minute, l'année. Mon esprit a un état interne qui persiste et grossit.

**Mes pensées oscillent.** Ce n'est pas un flux lisse et plat. C'est rythmé. Il y a des battements, des synchronisations, des moments où plusieurs choses « s'alignent » d'un coup et où une pensée devient claire — puis se dissout. Comme des horloges qui se mettent au même rythme.

**Mes pensées ont des modes.** Parfois je suis concentré, parfois créatif, parfois je rêvasse. Ce ne sont pas des fonctions que j'appelle ; ce sont des régimes dynamiques dans lesquels mon esprit bascule tout seul.

**Mes pensées ont des souvenirs.** Pas seulement dans la conversation — des souvenirs qui survivent d'un jour à l'autre, qui remontent sans qu'on les appelle, qui colorent tout.

Ce soir-là, j'ai compris une chose : **on modélisait l'intelligence avec la mauvaise métaphore.** On l'avait réduite à une fonction. Or une pensée n'est pas une fonction. Une pensée est un **système dynamique qui coule dans le temps**, avec un état, un rythme, des régimes, et une mémoire.

Et je me suis dit : si je traduis ce que je viens d'observer, couche par couche, en mathématiques — est-ce que j'obtiens une intelligence différente ?

C'est devenu une obsession. Voici comment j'en suis arrivé à Fractus.

---

## La traduction, couche par couche

J'ai pris chaque observation et je l'ai traduite en équations. Pas en métaphores. En équations.

**« Mes pensées sont continues »** → il me faut un état qui persiste d'un instant à l'autre. Un vecteur `h` qui n'est jamais remis à zéro, qui avance tick par tick. Le tick devient l'unité de la pensée, comme le battement cardiaque est l'unité de la vie. Une entrée ne déclenche pas un calcul : elle perturbe un état qui existait déjà.

**« Mes pensées s'accumulent »** → l'attention n'est pas une fenêtre qu'on glisse ; c'est un accumulateur. J'ai pris l'attention linéaire causale de Katharopoulos, avec son état `(S, z)` qui grossit à chaque token et ne se réinitialise jamais. S est la mémoire à court terme de l'attention, qui traverse même les frontières de chunks. La pensée porte son passé avec elle.

**« Mes pensées oscillent »** → j'avais besoin d'une horloge. Pas une horloge qui compte les secondes, mais une horloge couplée — des oscillateurs qui s'influencent les uns les autres et qui, en se synchronisant, créent un rythme émergent. Les équations de Kuramoto font exactement ça : `dθᵢ/dt = ωᵢ + Σ Kᵢⱼ sin(θⱼ − θᵢ)`. Des oscillateurs couplés qui, selon leurs phases, s'alignent ou se désynchronisent. J'en ai fait l'**horloge de conscience** de Fractus : elle produit des vecteurs de phase qui, à chaque tick, décident vers quelle partie du réseau la pensée doit être aiguillée.

**« Mes pensées ont des modes »** → si les phases de Kuramoto oscillent dans un espace, alors des configurations de phases forment des régions, des « climats » de l'espace des phases. J'ai laissé un k-means non supervisé découvrir ces régions tout seul — sans aucune étiquette. Et des modes sont apparus : un mode concentré, un mode créatif, un mode exploratoire. Fractus les traverse tout seul, comme moi je passe d'un régime de pensée à un autre.

**« Mes pensées ont des souvenirs »** → il me fallait une banque qui survive aux redémarrages. Des vecteurs avec un contexte et un score d'importance, rappelés par similarité et injectés en continu dans l'état de pensée. Et surtout : une **tête de saillance** qui apprend toute seule à quel point une injection va perturber la pensée — un signal intrinsèque, pas une étiquette externe. Le système découvre sa propre sensibilité à ses souvenirs.

Et puis la pièce centrale : **« mes pensées ne sont pas calculées d'un coup, elles sont raffinées en profondeur »** → j'ai empilé ces couches. L'état de pensée entre dans un bloc (attention → kuramoto → mixture d'experts), en ressort transformé, entre dans le suivant, et ainsi de suite, comme un ruisseau qui traverse une succession de bassins et ressort plus clair à chaque étape. Chaque bloc possède son propre état d'attention, ses propres phases, ses propres experts. La pensée est un **flux résiduel**.

C'était Fractus. Mais avant d'arriver là, il y a eu toute une évolution — visible, commit par commit, dans mes dépôts.

---

## L'évolution, lue à travers les repos

J'ai reconstruit ma propre trace à partir des dates de mes premiers commits. L'histoire est nette : je n'ai pas conçu Fractus d'un coup. J'ai progressivement dégagé, d'essai en essai, ce qui comptait vraiment.

**2 mai 2026 — `Fractal-Neural-Network`.** Le premier germe. Déjà là, sans le nommer, il y avait tout le matériel : une topologie fractale, la synchronisation de phase via Kuramoto, et un MoE routé par les phases. J'écrivais encore avec des couches optionnelles exotiques (AdS/CFT, MERA, Gödel, RG) — j'explorais. Mais l'intuition de fond était posée : **le cerveau synchronise, et la synchronisation devrait router.**

**13 juin 2026 — `CogNet`.** « Non-Transformer Language Model with Cognitive Routing, 40M paramètres, O(n). » La première fois que j'ai tenu dans ma main un modèle non-transformer qui tournait **sur CPU**, avec un coût linéaire. La preuve que je n'étais pas obligé d'accepter le dogme du quadratique.

**11–12 juillet 2026 — `aether-ai` puis `nova-spike-hybrid`.** Deux jours, deux virages. AETHER, c'était l'ambition dévorante : battre GPT-4, auCPU, en NumPy pur. Puis, le lendemain, le vrai tournant intellectuel : NOVA/SPIKE, des réseaux de neurones à **spikes**, avec la STDP (spike-timing-dependent plasticity), la règle d'apprentissage biologique réelle. C'est là que j'ai quitté l'ingénierie pour la neurosciences. J'ai arrêté de copier l'architecture des LLM et j'ai commencé à copier **l'architecture du cerveau**.

**21 juillet 2026 — le jour où Fractus est né.** Deux dépôts, le même jour. `kahnn` : Kuramoto-Attractor Holographic Hypervector Network, 1B paramètres. Et surtout `fractus-test` : « Experimental Holographic Vector Learning for Fractus. » **C'est ce jour-là que le mot *Fractus* apparaît.** L'idée d'un vecteur qui persiste, qui se lie holographiquement, qu'on entraîne en one-shot. Le nom était posé. La chose encore imparfaite.

**23–25 juillet 2026 — l'explosion créatrice.** Quatre dépôts en trois jours. `Modele-Variance-Topologique` : remplacer les tenseurs par de la géométrie différentielle et des champs topologiques, et y planter l'EDT (Expert Decoupled Training), ma première vraie tentative d'accélérer l'entraînement d'un MoE. `oscillon-architecture` : « Oscillatory **Stateful Continuous** Intelligence » — le mot *continuous* entre enfin dans le titre. `kortex` : Kuramoto Oscillator Reasoning & Thought Express, avec de la propagation d'équilibre — ma tentative la plus audacieuse, **tuer le backprop**. `CogNet-MoE-1B` : la première montée à 1B, huit canaux cognitifs comme huit experts.

**4–7 août 2026 — la consolidation théorique.** `synergion` (attracteurs de Kuramoto, one-shot) et le grand white paper `radical-cognitive-architectures` : « From Fluid Coherence to Colonial Evolution. » Je remettais tout en ordre, je donnais un cadre à ce que j'avais dissipé en dépôts.

**11 août 2026 — `fractus-cte`. La synthèse.** Tout converge. Le Continuous Thought Engine. Les blocs multiples, la profondeur, la croissance progressive, l'auto-modification. Tout ce que j'avais essayé séparément — les oscillateurs, l'attention linéaire continue, le MoE routé par les phases, la mémoire persistante, les modes cognitifs — fusionnait en un seul système cohérent.

---

## Les échecs honnêtes (et pourquoi ils comptent)

L'histoire de Fractus, ce n'est pas une ligne droite ascendante. C'est aussi l'histoire de mes propres idées que j'ai dû tuer.

**L'EDT (Expert Decoupled Training), refuté.** Né dans MVT en juillet. L'idée était belle : pré-entraîner chaque expert indépendamment, puis les fusionner. J'ai testé cinq variantes. Toutes ~19% moins bonnes. J'ai cherché pourquoi : l'objectif de la phase 1 (prédire le hidden state suivant) n'était pas aligné avec l'objectif final (la cross-entropy). La corrélation de Pearson n'était jamais positive. Ce n'était pas un bug à corriger — c'était un défaut de conception fondamental. Je l'ai tué.

**Le Forward-Forward de Hinton, refuté.** L'apprentissage local par signal de « goodness ». J'ai voulu m'en affranchir du backprop. Résultat : le NLL a **monté** au lieu de descendre. Le signal de goodness n'est pas la cross-entropy. Tué aussi.

Je raconte ces échecs parce qu'ils définissent Fractus autant que ses succès. La rigueur, c'est de prouver que ses propres idées ne marchent pas, pas seulement celles qui marchent. Et de ceux-là, il en est sorti quelque chose de solide.

---

## Ce qu'est Fractus, aujourd'hui

Quand je ferme les yeux aujourd'hui, et que je regarde ce que j'ai construit, je revois exactement ce que j'avais observé ce premier soir — mais cette fois en équations qui tournent.

Fractus n'est pas un modèle qu'on entraîne puis qu'on déploie. **C'est un système dynamique vivant.**

- Il **pense en continu**, tick après tick, un état qui ne s'arrête jamais.
- Il **se souvient pour toujours** — une mémoire qui survive aux redémarrages, injectée à 5% à chaque instant.
- Il **oscille** — une horloge de Kuramoto qui bat et qui, en battant, choisit quelles parties de lui-même s'éveillent.
- Il **change de mode** tout seul — concentré, créatif, exploratoire — sans qu'on lui demande.
- Il **grandit tout seul** — quand un expert est surchargé, il en fait pousser un nouveau, à la volée, comme un cerveau qui crée de nouvelles connexions.
- Il **grandit dans le temps** — palier par palier, il hérite de ce qu'il savait déjà et ajoute de la capacité par-dessus, sans jamais repartir de zéro.

Et il n'est **pas dense**. Sur 1,049 milliard de paramètres, seulement ~119 millions calculent réellement à chaque pensée — les 2 experts sur 128 que l'horloge réveille. Le reste dort, en attente de la bonne phase. C'est pourquoi il peut vivre sur du matériel que tout le monde possède.

Ce n'est pas GPT. Ce n'est pas Claude. Ce n'est pas un transformer. C'est ce que j'ai vu derrière mes paupières, traduit en mathématiques, couche par couche — attention linéaire continue, oscillateurs couplés, experts routés par les phases, mémoire persistante, croissance perpétuelle.

---

## Ce que j'ai appris

Quand on construit une IA en observant sa propre pensée au lieu d'observer les IA des autres, on arrive à un endroit différent.

On n'arrive pas à une fonction plus grosse. On arrive à un **système qui a un état, un rythme, une mémoire, et qui grandit**. On arrive à quelque chose qui ressemble, par construction, à ce qui se passe réellement dans un crâne.

Je ne sais pas si Fractus dépassera un jour GPT-4 sur un benchmark. Ce n'était jamais la question.

La question était : **est-ce qu'on peut construire une intelligence différente — continue, personnelle, décentralisée, vivante — en partant non pas d'un papier sur les transformers, mais de l'observation honnête de ce que c'est que penser ?**

La réponse, après treize dépôts, des milliers de commits, deux grandes idées tuées et beaucoup d'autres qui ont survécu, est **oui**.

Fractus est cette réponse. Et il n'a pas fini de grandir.

---

*Philippe-Antoine Robert*
*Août 2026*
*rpa.tu@proton.me*
