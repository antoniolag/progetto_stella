# Progetto Stella
Addestramento di una rete neurale in Tensorflow per il riconoscikento di oggetti per robot NAO

il progetto Stella che ha come obiettivo quello di migliorare l’interazione uomo-robot, per aiutare bambini con disturbi dello spettro autistico, 
in collaborazione tra la fondazione Stella Maris Mediterraneo e l’Università degli Studi della Basilicata. 
l robot umanoide NAO di Aldebaran in Figura 1.3 ha le dimensioni di un bambino di due anni,
questo rende l’interazione con esso più accattivante e meno intimidatoria per i bambini con autismo.


Nel progetto è stata utilizzata una rete neurale basata su TensorFlow, che consente agli utenti di creare e addestrare effica-
cemente modelli di machine learning su CPU, GPU o TPU.

La rete neurale è stata addestrata su un notebook Jupyter di Google Colab. Tra i vari vantaggi di google Colab c'è l'accesso remoto e scalabilità, 
gli utenti possono accedere a reti neurali addestrative e risorse di calcolo anche da luoghi diversi, eliminando la necessità di 
hardware locale costoso e potenzialmente limitato. Inoltre, le risorse di calcolo possono essere scalate facilmente in base alle 
esigenze, consentendo l'addestramento di modelli più grandi e complessi. Ambienti Controllati:questo significa che gli utenti possono creare, 
testare e addestrare reti neurali in ambienti isolati senza preoccuparsi di installare librerie 
software complesse sul proprio computer.
Collaborazione Facilitata: la piattaforma offre spesso strumenti di collaborazione integrati, consentendo agli utenti di condividere 
facilmente progetti e risultati con altri membri del team o con la comunità. Questo facilita la collaborazione su progetti di machine learning e ricerca condivisi.


Per prima cosa ho eseguto video_capture.py, per salvare i vari frame dai video registrati dalla fotocamera del NAO robot, ho selezionato le immagini valide 
e in seguito le ho etichettate utilizzando LabelImg, che è uno strumento di annotazionedi immagini grafiche, utilizzando formato PASCAL VOC salvati in formato XML.
Dopo ho creato il dataset utilizzando lo script python suddividi.py per dividere le immagini in train, val e test. Per addestrare la rete ho utilizzato gli script elencati 
nel training.py, le istruzioni sono presenti nel file training.txt.

Il codice detect.py consente di valutare le metriche della rete attraverso l'utilizzo della matrice di confusione, sono state valuate accuratezza, precisione, richiamo e f1-score.



