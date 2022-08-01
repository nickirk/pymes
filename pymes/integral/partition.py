


def part_2_body_int(no, t_V_pqrs):
    dict_t_V = {}
    t_V_abci = t_V_pqrs[no:, no:, no:, :no]
    dict_t_V['abci'] = t_V_abci
    t_V_iabj = t_V_pqrs[:no, no:, no:, :no]
    dict_t_V['iabj'] = t_V_iabj
    t_V_iajk = t_V_pqrs[:no, no:, :no, :no]
    dict_t_V['iajk'] = t_V_iajk
    t_V_aijk = t_V_pqrs[no:, :no, :no, :no]
    dict_t_V['aijk'] = t_V_aijk
    t_V_klij = t_V_pqrs[:no, :no, :no, :no]
    dict_t_V['klij'] = t_V_klij
    t_V_aibj = t_V_pqrs[no:, :no, no:, :no]
    dict_t_V['aibj'] = t_V_aibj
    t_V_ijak = t_V_pqrs[:no, :no, no:, :no]
    dict_t_V['ijak'] = t_V_ijak
    t_V_abic = t_V_pqrs[no:, no:, :no, no:]
    dict_t_V['abic'] = t_V_abic
    t_V_iajb = t_V_pqrs[:no, no:, :no, no:]
    dict_t_V['iajb'] = t_V_iajb
    t_V_abcd = t_V_pqrs[no:, no:, no:, no:]
    dict_t_V['abcd'] = t_V_abcd
    t_V_iabc = t_V_pqrs[:no, no:, no:, no:]
    dict_t_V['iabc'] = t_V_iabc
    t_V_aijb = t_V_pqrs[no:, :no, :no, no:]
    dict_t_V['aijb'] = t_V_aijb
    t_V_ijka = t_V_pqrs[:no, :no, :no, no:]
    dict_t_V['ijka'] = t_V_ijka
    t_V_aibc = t_V_pqrs[no:, :no, no:, no:]
    dict_t_V['aibc'] = t_V_aibc
    t_V_ijab = t_V_pqrs[:no, :no, no:, no:]
    dict_t_V['ijab'] = t_V_ijab
    t_V_abij = t_V_pqrs[no:, no:, :no, :no]
    dict_t_V['abij'] = t_V_abij

    return dict_t_V
