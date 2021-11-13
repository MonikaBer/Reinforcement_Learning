#!/bin/bash

# skrypt usuwa wybrane pakiety. Możliwe jest, że należy usunąć numer wersji danego pakietu albo ją zaktualizować dzięki pip freeze > tmp.txt

pip uninstall AutoROM==0.4.2 AutoROM.accept-rom-license==0.4.2 
pip uninstall dm-acme==0.2.2 dm-control==0.0.403778684 dm-env==1.5 dm-launchpad-nightly==0.3.0.dev20211113 dm-reverb==0.5.0 dm-tree==0.1.6 
pip uninstall gym==0.21.0 
pip uninstall tensorboard==2.6.0 
pip uninstall tensorflow==2.6.2 tensorflow-datasets==4.4.0 tensorflow-estimator==2.6.0 tensorflow-io-gcs-filesystem==0.22.0 
pip uninstall tensorflow-metadata==1.4.0 tensorflow-probability==0.14.1