import tenseal as ts


def get_he_context():
    # controls precision of the fractional part
    bits_scale = 40
    # create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            60,
            bits_scale,
            bits_scale,
            60,
        ],
    )
    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context
