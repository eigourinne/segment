# yolo-based picture segment system for medical

> 基于yolo的医学图像分割系统

- thanks to ultralytics, kaggle, streamlit

- made by maou

- email:[onigami@qq.com]/[rinneeigou@gmail.com]

## requirement

- cuda(if not, it may happen *2000 years later*)
- nccl(if you use package manager, it will be automatively download while installing cuda or torch-cuda)
- pytorch-cuda(pytorch cpu is really slow)
- torchvivion-cuda(same reason)
- streamlit(usually gradio is more popular, but in archlinux's aur, gradio is facing cycle-requirement problems)
- customs-tkinter(test.py require it)

## architecture

- to_yolo.py
- data.yaml
- train.py
- test.py
- metrics.py
- gui.py

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣆⠀⠀⢸⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠈⣿⡄⠈⣿⣿⣿⣦⡀⢸⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⠣⠀⠈⢿⡟⢻⣷⣿⢿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⡿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠙⢿⣿⣷⡀⠀⠁⠀⠈⠇⠀⢿⣿⣸⣿⣿⣿
⣀⣀⢸⣤⣄⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⠿⠛⠁⢺⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣷⡀⠈⠻⣿⣿⡀⠀⢠⣤⣼⣤⠆⠙⢿⣿⣿⣿
⣿⣿⠀⣿⣿⠃⠀⠀⠀⠀⢠⣿⣿⣿⣿⣷⡀⠀⣠⣽⣿⣿⣿⣿⣿⡿⠋⢸⡿⢸⣿⣿⣿⣿⣄⠀⠙⢿⣇⠀⠀⠿⣍⠿⠀⢀⣿⣿⣿⣿
⢿⣿⠀⠻⢿⠀⠀⠀⠀⠀⣾⣿⣿⣿⢫⣿⣇⣼⣿⠻⣿⣿⣿⣿⡿⠁⠀⢸⡇⢸⣿⣿⣿⣿⣿⣦⡀⠀⠻⠀⠀⠘⠀⠘⠀⣸⣿⣿⣿⣿
⠀⢻⠀⠀⢸⠀⠀⠀⠀⡸⢻⣿⣿⣧⢸⣿⣿⣯⡀⢀⣻⡿⢱⡟⠀⠀⠀⣸⡁⢘⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿
⠀⣼⠀⠀⢸⠀⠀⠀⢠⠃⢸⣿⣿⣿⣿⢸⣿⣿⣇⣿⣿⠃⠸⠀⠀⠀⠀⠘⠀⠀⠇⢿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿
⠀⢿⠀⠀⢸⠀⠀⢠⠃⠀⢸⣿⣿⣿⣿⣆⣿⣿⣿⣿⡇⠀⠰⣦⣄⡀⢀⣐⠢⠀⠀⠀⠹⣿⠿⠟⠁⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿
⠀⢾⠀⠀⢸⠀⢀⠂⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⠀⠈⠙⠿⠿⠿⠁⠀⠀⠀⠀⠀⠀⠴⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⣿⡟⠙⢯⠻⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⢿⡆⠀⠀⠀⠀⠀⠀⢸⣿⣏⣿⣿⣿⣿⣿
⠀⠀⠀⠀⣰⠃⠀⠀⣼⣿⣿⣿⣿⣿⣿⠟⠀⠀⠀⠀⠙⠣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠘⠋⢃⣿⣿⣿⣿⣿
⣠⣤⣶⣶⣿⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⡄⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⣠⣾⣿⣿⠄⠀⠀⠀⠀⠀⠃⠀⢻⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⢄⠀⠀⠀⠀⠀⠀⢀⣀⣤⣶⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠈⠐⡨⠒⠐⠂⢹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣤⣀⣀⣀⣠⣤⣴⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣶⣦⣤⡂⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣤⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⡏⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿

## surprise

- genshin impact NB!
- arknights NB!
- honkai impact 3 NB!
- honkai impact 2 NB!
- honkai: star rail NB!
- zenless zero zone NB!
- arknights:endfield NB!
- azur lane NB!
- blue archive NB!
- wuthering waves NB!
- legands of the three kindoms NB!
- bang dream! NB!
- d4dj NB!
- revue starlight NB!
- maj_soul NB!
- masterduel NB!
- terraria NB!
- minecraft NB!
- slay the spire NB!
- slay the spire 2 NB!
- atcoder NB!
- nowcoder NB!
- codeforces NB!
