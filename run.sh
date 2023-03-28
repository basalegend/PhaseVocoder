# Проверяем количества аргументов  командной строки
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input.wav> <output.wav> <time_stretch_ratio>"
  exit 1
fi

# Запускаем main.py с заданными аргументами
python3 main.py "$1" "$2" "$3"
