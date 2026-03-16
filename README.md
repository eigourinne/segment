# yolo-based segment system for medical

> 基于yolo的医学图像分割系统

- thanks to ultralytics, kaggle, streamlit

- made by maou

- email:[onigami@qq.com]/[rinneeigou@gmail.com]

> genshin impact NB!
> arknights NB!
> honkai impact 3 NB!
> honkai impact 2 NB!
> honkai: star rail NB!
> zenless zero zone NB!
> arknights:endfield NB!
> blue archive NB!
> wuthering waves NB!
> three kindoms kill NB!
> bang dream! NB!
> d4dj NB!
> revue starlight NB!
> maj_soul NB!
> masterduel NB!
> terraria NB!
> minecraft NB!
> slay the spire NB!
> slay the spire 2 NB!
> atcoder NB!
> nowcoder NB!
> codeforces NB!

## architecture

- to_yolo.py
- data.yaml
- main.py
- test.py
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

Copyright (C)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
