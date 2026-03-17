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

- to_yolo.py -> just used to convert (images, masks) to yolo's shape
- data.yaml -> adjust it as what your need
- main.py -> dont't forget to download model(in my ultralytics provided by archlinux, it just offer v8's, so I mannally donwload v11n-seg for this project, if you wanna solve objection-problem, use v11n/v26n etc.)
- test.py -> simple tkinter interface
- gui.py -> use streamlit to build the back end, then display it on the front end

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

Copyright (C)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
