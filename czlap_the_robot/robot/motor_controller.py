import numpy as np


class Controller:
    def __init__(self, Kp, Ki, Kd, N, sample_time, saturation=(-np.inf,np.inf)):
        """
        
        Discrete PID controller with filtering cooficient N
                              1         N * Kd
        G(s) = Kp   +   Ki * ---   +   ---------
                              s         1 + N/s
        Discretized using bilinear transform where:
             2     z - 1
        s = --- * -------
             Ts    z + 1

        After discretization trasnsfer functions is as follows

                          Ts      z+1                     N * (z-1)
        G(z) = Kp + Ki * ----  * ----- + Kd *  ---------------------------------
                           2      z-1           (1 + N*Ts/2) * z + N * Ts/2 - 1


        After clearing z values

                 ((2*N*Ts + 4)*Kp + (N*Ts^2 + 2*Ts)*Ki + 4*N*Kd) * z^2 + (-8*Kp + 2*N*Ts^2*Ki - 8*N*Kd) * z + (4 - 2*N*Ts)*Kp + (N*Ts^2 - 2*Ts)*Ki + 4*N*Kd
        G(z) =  --------------------------------------------------------------------------------------------------------------------------------------------
                                                    (2*N*Ts + 4)*z^2 - 8*z - 2*N*Ts + 4

        """

        self._Ts = sample_time
        self._N  = N
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._saturation = saturation

        self._last_errors  = np.zeros(3)
        self._last_outputs = np.zeros(2)

        # calculate numerator and denominator cooficients
        b0 = (4 + 2*self._N*self._Ts) * self._Kp + (self._N*self._Ts**2 + 2*self._Ts) * self._Ki + (4*self._N) * self._Kd
        b1 =                     (-8) * self._Kp +            (2*self._N*self._Ts**2) * self._Ki - (8*self._N) * self._Kd
        b2 = (4 - 2*self._N*self._Ts) * self._Kp + (self._N*self._Ts**2 - 2*self._Ts) * self._Ki + (4*self._N) * self._Kd

        a0 = (4 + 2*self._N*self._Ts)
        a1 =                     (-8)
        a2 = (4 - 2*self._N*self._Ts)

        self._cooficients =  np.array([-a1, -a2, b0, b1, b2]) / a0
        self._cooficients = self._cooficients.reshape(self._cooficients.shape[0], 1)



    def __call__(self, error):
        self._last_errors[1:] = self._last_errors[:2]
        self._last_errors[0] = error

        dynamics = np.concatenate((self._last_outputs, self._last_errors), axis=0)
        
        control_signal = np.clip((dynamics @ self._cooficients)[0], self._saturation[0], self._saturation[1])

        self._last_outputs = np.array([control_signal, self._last_outputs[0]])

        return control_signal