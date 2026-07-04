from cysignals.alarm import alarm, AlarmInterrupt, cancel_alarm
import requests
from subprocess import Popen, PIPE
import re

_parse_status_re = re.compile(
        r'Using B1=(\d+), B2=(\d+), polynomial ([^,]+), sigma=(\d+)')

_found_input_re = re.compile('Found input number N')

_found_factor_re = re.compile(
    r'Found (?P<primality>.*) factor of [\s]*(?P<digits>\d+) digits: (?P<factor>\d+)')

_found_cofactor_re = re.compile(
    r'(?P<primality>.*) cofactor (?P<cofactor>\d+) has [\s]*(?P<digits>\d+) digits')

def _parse_output(n, out):
    out_lines = out.lstrip().splitlines()
    if not out_lines[0].startswith('GMP-ECM'):
        raise ValueError('invalid output')
    result = []
    for line in out_lines:
        # print('parsing line >>{0}<<'.format(line))
        m = _parse_status_re.match(line)
        if m is not None:
            group = m.groups()
            _last_params = {'B1': group[0], 'B2': group[1],
                                    'poly': group[2], 'sigma': group[3]}
            continue
        m = _found_input_re.match(line)
        if m is not None:
            return [(n, True)]
        m = _found_factor_re.match(line)
        if m is not None:
            factor = m.group('factor')
            primality = m.group('primality')
            assert primality in ['prime', 'composite', 'probable prime']
            result += [(ZZ(factor), primality != 'composite')]
            continue  # cofactor on the next line
        m = _found_cofactor_re.match(line)
        if m is not None:
            cofactor = m.group('cofactor')
            primality = m.group('primality')
            assert primality in ['Prime', 'Composite', 'Probable prime']
            result += [(ZZ(cofactor), primality != 'Composite')]
            # assert len(result) == 2
            return result
    raise ValueError('failed to parse ECM output')

def try_factor(n):
    cmd = '/usr/bin/ecm -c 1000000000 -I 1 -one 2000'
    todo = [n]
    facs = []
    try:
        alarm(3)
        while todo:
            todo = sorted(todo)
            u = todo.pop(0)
            if u.bit_length() < 80:
                facs.append(u)
                continue
            p = Popen(cmd.split(), stdout=PIPE, stdin=PIPE, stderr=PIPE, encoding='latin-1')
            out, err = p.communicate(input=str(u))
            result = _parse_output(u, out)
            for f, is_p in result:
                if is_p:
                    facs.append(f)
                else:
                    todo.append(f)
    except AlarmInterrupt:
        p.kill()
        pass
    except ValueError:
        print(u)
    finally:
        cancel_alarm()
    return facs

def factordb(n):
    url = f"https://factordb.com/api?query={n}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        facs = data['factors']
    return [(int(f[0]), int(f[1])) for f in facs]

def flatter(M):
    from subprocess import check_output
    from re import findall
    z = "[[" + "]\n[".join(" ".join(map(str, row)) for row in M) + "]]"
    ret = check_output(["flatter"], input=z.encode())
    return matrix(M.nrows(), M.ncols(), map(int, findall(b"-?\\d+", ret)))

def modular_hermite_form(M):
    ID = identity_matrix(ZZ, M.ncols())
    modulus = M.base_ring().cardinality()
    rows = M.rows()
    basis = []
    for i in tqdm(range(M.ncols())):
        b = rows[0]
        bi = ZZ(b[i])
        if bi != 0:
            g, u, v = xgcd(bi, modulus)
            b = u * b
            bi = ZZ(b[i])
        for r in rows[1:]:
            ri = ZZ(r[i])
            if ri == 0:
                continue
            if bi == 0:
                b = r
                bi = ri
                continue
            if ri % bi == 0:
                continue
            else:
                g, u, v = xgcd(bi, ri)
                b = u * b + v * r
                bi = ZZ(b[i])
                assert bi == g
        if bi == 0:
            basis.append(ID.row(i)*modulus)
            continue
        g, u, v = xgcd(bi, modulus)
        b = u * b
        bi = ZZ(b[i])
        basis.append(b)
        new_rows = []
        for r in rows:
            c = ZZ(r[i]) // bi
            new_r = r - c * b
            if not new_r.is_zero():
                new_rows.append(new_r)
        rows = new_rows
        if len(rows) == 0:
            rows.append(zero_vector(ZZ, M.ncols()))
    return matrix(ZZ, basis)

def babai_cvp(B, t, perform_reduction=True):
    if perform_reduction:
        B = B.LLL(delta=0.75)

    G = B.gram_schmidt()[0]
    b = t
    for i in reversed(range(B.nrows())):
        c = ((b * G[i]) / (G[i] * G[i])).round()
        b -= c * B[i]

    return t - b