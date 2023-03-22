import copy #JEC

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array'):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    stack = backend.stack

    # Define lists for output.
    out_S_0, out_S_1, out_S_1_JEC, out_S_2 = [], [], [], []

    U_r = pad(x)

    U_0_c = rfft(U_r)

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi['levels'][0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)


    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)

        print(f"JEC 1: n1:{n1},j1:{j1},t:{theta1}: U1c", U_1_c.shape)

        #JEC
        U_1_c_sav = copy.deepcopy(U_1_c)
        #U_1_c_sav= subsample_fourier(U_1_c_sav, k=2 ** (J - j1))
        print(f"JEC 1b: n1:{n1},j1:{j1},t:{theta1}: U1csav", U_1_c_sav.shape)
        #U_1_c_sav = unpad(U_1_c_sav)
        out_S_1_JEC.append({'coef_jec':U_1_c_sav,
                            'j': (j1,),
                            'n': (n1,),
                            'theta': (theta1,)
                            })

        U_1_c = modulus(U_1_c)
        print(f"JEC 2: n1:{n1},j1:{j1},t:{theta1}: U1c", U_1_c.shape)
        U_1_c = rfft(U_1_c)
        print(f"JEC 3: n1:{n1},j1:{j1},t:{theta1}: U1c", U_1_c.shape)

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi['levels'][j1])
        print(f"JEC 4: n1:{n1},j1:{j1},t:{theta1}: S1c", S_1_c.shape)

        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))
        print(f"JEC 5: n1:{n1},j1:{j1},t:{theta1}: S1c", S_1_c.shape)

        S_1_r = irfft(S_1_c)
        print(f"JEC 6: n1:{n1},j1:{j1},t:{theta1}: S1r", S_1_r.shape)

        S_1_r = unpad(S_1_r)
        print(f"JEC 7: n1:{n1},j1:{j1},t:{theta1}: S1r", S_1_r.shape)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'n': (n1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi['levels'][j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r)

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'n': (n1, n2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)
    out_S.extend(out_S_1_JEC) # JEC

    if out_type == 'array':
        out_S = stack([x['coef'] for x in out_S])

    return out_S


__all__ = ['scattering2d']
