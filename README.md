# conta veicoli da webcam di viaggiaretrentino.it

Una semplice applicazione che attraverso [pytorch](https://pytorch.org/) e [yolov5](https://pytorch.org/hub/ultralytics_yolov5/) conta il numero di veicoli che si vedono nelle [webcam](https://vit.trilogis.it/webcam/) di [viaggiareintrentino.it](https://www.viaggiareintrentino.it/#/index/it)

Lo script viene eseguito da una github action ogni cinque minuti e produce:
- le immagini delle webcam con sovraimpressi gli oggetti che ha individuato (ed un indice di valutazione)
- un file csv con le ultime informazioni raccolte
- lo storico di tutte le rilevazioni svolte

Vengono considerati i veicoli di tipo: car, truck, bus e train e con indice di valutazione superiore a 0.5.


![](https://raw.githubusercontent.com/napo/veicoliviaggiareintrentino/main/examples/example_cam50.jpg)

Questioni aperte:
- mancando l'illuminazione le webcam non sono in grado di individuare il numero di auto di notte
- ~vengono calcolate anche le auto nei parcheggi~


