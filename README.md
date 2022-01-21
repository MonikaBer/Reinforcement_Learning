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

## Dokumentacja
Wstępna:
[v1](https://demo.hedgedoc.org/CY_vyFK-R8u1ZieGybwuog?both#)

Końcowa:
[v2](https://demo.hedgedoc.org/eRSqFNRPT9Wp1qMolpwIIQ#)
