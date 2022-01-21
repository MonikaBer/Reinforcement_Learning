# Reinforcement_Learning

## Przygotowanie środowiska (dla Linux Ubuntu 20.04)
W katalogu głównym projektu:
- zainstalować Pythona w wersji 3.8 oraz pip
- zainstalować narzędzie virtualenv
- utworzyć i aktywować środowisko wirtualne
- zainstalować wymagane biblioteki

```
#instalacja Python 3.8
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8
python3.8 --version

#instalacja pip
sudo apt install python3.8-pip
pip3.8 -V

#instalacja virtualenva
pip3.8 install virtualenv

#utworzyć środowisko wirtualne
python3.8 -m venv venv

#aktywacja środowiska wirtualnego
source venv/bin/activate

#instalacja bibliotek z requirements.txt
pip3.8 install -r requirements.txt
```

## Uruchamianie własnych eksperymentów
Służy do tego skrypt _**main.py**_:
```
python3.8 src/main.py --help
```
Argumenty wywołania (wraz z domyślnymi wartościami - jeśli są):

Wymagane:

- --alg {dqn,impala} (typ algorytmu)

Opcjonalne:

- --num_episodes NUM_EPISODES (max liczba epizodów gry podczas treningu) (domyślnie 100)
- --max_steps MAX_STEPS (max liczba kroków gry podczas treningu) (domyślnie 45000)
- --batch_size BATCH_SIZE (rozmiar batcha) (domyślnie: dla DQN - 128, dla IMPALA - 16)
- --gpu {0,1} (uruchomienie na GPU) (domyślnie 1 - tak)
- --save_video {0,1} (zapisanie pliku video z ewaluacji modelu) (domyślnie 0 - nie)
- --save_csv {0,1} (zapisanie pliku csv z ewaluacji modelu) (domyślnie 0 - nie)
- --video_name VIDEO_NAME (nazwa pliku video z ewaluacji modelu) (domyślnie 'temp.mp4')
- --collect_frames COLLECT_FRAMES (liczba kroków ewaluacji modelu) (domyślnie 7500)
- --lr LR (learning rate) (domyślnie 1e-3)
- --discount DISCOUNT (dyskonto) (domyślnie 0.99)
- --target_update_period TARGET_UPDATE_PERIOD (tylko dla DQN, docelowa liczba kroków przed uaktualnieniem wag w docelowej sieci, z której korzysta DQN) (domyślnie 100)
- --entropy_cost ENTROPY_COST (tylko dla IMPALA, koszt entropii) (domyślnie 0.01)

Dodatkowe uwagi:

- pliki _**csv**_ są zapisywane w katalogu _*csv/*_
- pliki _**video**_ są zapisywane w katalogu _*video/*_

Przykładowe uruchomienie treningu DQN:
```
python3.8 src/main.py --alg dqn --num_episodes 500 --max_steps 10000 --lr 1e-5 --discount 0.95 --update_target_period 300
```

Przykładowe uruchomienie treningu IMPALI:
```
python3.8 src/main.py --alg impala --num_episodes 500 --max_steps 10000 --lr 1e-5 --discount 0.95 --entropy_cost 0.001
```

## Uruchamianie skryptów odpalających automatycznie kolejne eksperymenty
Skrypt dla DQN:
```
./src/scripts/train_dqn.sh
```

Skrypt dla IMPALA:
```
./src/scripts/train_impala.sh
```

## Analiza wyników eksperymentów
Służy temu skrypt _**csv_analysing.py**_:
```
python3.8 src/csv_analysing.py --help
```

Wymagane argumenty:

- --alg {dqn,impala} (typ algorytmu)
- --path PATH (ścieżka do katalogu z plikami csv do analizy (pochodzących z ewaluacji nauczonych modeli))

Dodatkowa uwaga:

- wykresy sporządzone na podstawie plików csv są zapisywane na ścieżce _**plots/<ALG_TYPE>/**_ 

## Dokumentacja
Wstępna:
[v1](https://demo.hedgedoc.org/CY_vyFK-R8u1ZieGybwuog?both#)

Końcowa:
[v2](https://demo.hedgedoc.org/eRSqFNRPT9Wp1qMolpwIIQ#)
