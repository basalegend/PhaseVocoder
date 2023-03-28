import numpy as np
import librosa


class PhaseVocoder(object):

    def __init__(self, audio, rate):
        """
        Инициализация функтора PhaseVocoder.
        Параметры
        ----------
        audio : np.ndarray
            Входной сигнал
        rate : float
            Коэффициент сжатия/разжатия аудиосигнала;
            При rate > 1 аудиофайл растягивается, при 0 < rate <= 1 сжимается
        ----------
        Во время инициализации сразу происходит STFT(Кратковременное преобразование Фурье),
        которое заключается в том, что оно разбивает входной сигнал на сегменты,
        применяет к ним оконную функцию(по умолчанию Ханна) и затем применяет
        FFT(быстрое преобразование Фурье) (БПФ) к каждому сегменту для получения его спектрального представления.
        """
        self.transformed_audio = librosa.stft(audio)
        self.rate = rate

    def _phase_vocoder(self):
        """
        Реализация алгоритма phase_vocoder. Данный метод вызывается во время магического метода __call__.
        """

        # Так как в STFT используются параметры по умолчанию, а именно n_fft(длина сегмента)=2048,
        # а так же по алгоритму перекрытие составляет 75%, то длина сдвига будет 2048 / 4 = 51
        hop_length = 512

        # Создание массива временных отсчётов
        time_steps = np.arange(0, self.transformed_audio.shape[-1], 1 / self.rate)

        # Инициализация результирующего массива
        shape = list(self.transformed_audio.shape)
        shape[-1] = len(time_steps)
        output = np.zeros_like(self.transformed_audio, shape=shape)

        # Вычисление ожидаемого изменения фазы
        phi_advance = np.linspace(0, np.pi * hop_length, self.transformed_audio.shape[-2])

        # Сумма фаз, определяем первым отсчетом
        phase_acc = np.angle(self.transformed_audio[:, 0])

        # Добавление нулевых столбцов слева от массива
        padding = [(0, 0) for _ in self.transformed_audio.shape]
        padding[-1] = (0, 2)
        self.transformed_audio = np.pad(self.transformed_audio, padding, mode="constant")

        # Цикл по временным отсчетам
        for t, step in enumerate(time_steps):
            # берём по две соседние колонки
            columns = self.transformed_audio[:, int(step): int(step + 2)]

            # Коэффициент, используемый для вычисления взвешенного линейного интерполирования амплитуды
            # между двумя близлежащими блоками спектрограммы
            alpha = np.mod(step, 1.0)

            # Взвешенное среднее амплитуды между двумя блоками спектрограммы
            mag = (1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1])

            # Сохранение в результирующий массив спектра сигнала после его изменения
            output[:, t] = (np.cos(phase_acc) + 1j * np.sin(phase_acc)) * mag

            # Вычисление изменения фазы.
            delta_phase = np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance

            # Переводим в диапазон от -pi до pi
            delta_phase = delta_phase - 2.0 * np.pi * np.round(delta_phase / (2.0 * np.pi))

            # Суммируем изменения фазы
            phase_acc += phi_advance + delta_phase

        # Выполнение ISTFT(обратного кратковременного преобразования Фурье)
        output = librosa.istft(output)

        return output

    def __call__(self, *args, **kwargs):
        return self._phase_vocoder()
