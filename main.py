import librosa
import scipy.io.wavfile as wavfile
from PhaseVocoder import PhaseVocoder
import sys


def main():
    # распаковываем аргументы командной строки
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    rate = float(sys.argv[3])

    # загрузка аудиофайла, частота дискретизации собственная(sr=None)
    audio_input, sample_rate = librosa.load(input_filename, sr=None)

    # создаём функтор для обработки сигнала
    instance = PhaseVocoder(audio_input, rate)

    # обрабатываем аудиосигнал
    audio_output = instance()

    # сохранение результат в файл(из librosa убрали librosa.output.write_wav, поэтому записываем посредством scipy)
    wavfile.write(output_filename, sample_rate, audio_output)


if __name__ == "__main__":
    main()
