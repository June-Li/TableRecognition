import numpy as np
import string
import torch


class TableLabelDecode(object):
    """  """
    def __init__(self,
                 character_dict_path,
                 **kwargs):
        list_character, list_elem = self.load_char_elem_dict(character_dict_path)
        list_character = self.add_special_char(list_character)
        list_elem = self.add_special_char(list_elem)
        self.dict_character = {}
        self.dict_idx_character = {}
        for i, char in enumerate(list_character):
            self.dict_idx_character[i] = char
            self.dict_character[char] = i
        self.dict_elem = {}
        self.dict_idx_elem = {}
        for i, elem in enumerate(list_elem):
            self.dict_idx_elem[i] = elem
            self.dict_elem[elem] = i

    def load_char_elem_dict(self, character_dict_path):
        list_character = []
        list_elem = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            substr = lines[0].decode('utf-8').strip("\n").strip("\r\n").split("\t")
            character_num = int(substr[0])
            elem_num = int(substr[1])
            for cno in range(1, 1 + character_num):
                character = lines[cno].decode('utf-8').strip("\n").strip("\r\n")
                list_character.append(character)
            for eno in range(1 + character_num, 1 + character_num + elem_num):
                elem = lines[eno].decode('utf-8').strip("\n").strip("\r\n")
                list_elem.append(elem)
        return list_character, list_elem

    def add_special_char(self, list_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        list_character = [self.beg_str] + list_character + [self.end_str]
        return list_character

    def __call__(self, preds):
        structure_probs = preds['structure_probs']
        loc_preds = preds['loc_preds']
        if isinstance(structure_probs, torch.Tensor):
            structure_probs = structure_probs.numpy()
        if isinstance(loc_preds, torch.Tensor):
            loc_preds = loc_preds.numpy()
        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)
        structure_str, structure_pos, result_score_list, result_elem_idx_list = self.decode(structure_idx,
                                                                                            structure_probs, 'elem')
        res_html_code_list = []
        res_loc_list = []
        batch_num = len(structure_str)
        for bno in range(batch_num):
            res_loc = []
            for sno in range(len(structure_str[bno])):
                text = structure_str[bno][sno]
                if text in ['<td>', '<td']:
                    pos = structure_pos[bno][sno]
                    res_loc.append(loc_preds[bno, pos])
            res_html_code = ''.join(structure_str[bno])
            res_loc = np.array(res_loc)
            res_html_code_list.append(res_html_code)
            res_loc_list.append(res_loc)
        return {'res_html_code': res_html_code_list, 'res_loc': res_loc_list, 'res_score_list': result_score_list,
                'res_elem_idx_list': result_elem_idx_list, 'structure_str_list': structure_str}

    def decode(self, text_index, structure_probs, char_or_elem):
        """convert text-label into text-index.
        """
        if char_or_elem == "char":
            current_dict = self.dict_idx_character
        else:
            current_dict = self.dict_idx_elem
            ignored_tokens = self.get_ignored_tokens('elem')
            beg_idx, end_idx = ignored_tokens

        result_list = []
        result_pos_list = []
        result_score_list = []
        result_elem_idx_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            elem_pos_list = []
            elem_idx_list = []
            score_list = []
            for idx in range(len(text_index[batch_idx])):
                tmp_elem_idx = int(text_index[batch_idx][idx])
                if idx > 0 and tmp_elem_idx == end_idx:
                    break
                if tmp_elem_idx in ignored_tokens:
                    continue

                char_list.append(current_dict[tmp_elem_idx])
                elem_pos_list.append(idx)
                score_list.append(structure_probs[batch_idx, idx])
                elem_idx_list.append(tmp_elem_idx)
            result_list.append(char_list)
            result_pos_list.append(elem_pos_list)
            result_score_list.append(score_list)
            result_elem_idx_list.append(elem_idx_list)
        return result_list, result_pos_list, result_score_list, result_elem_idx_list

    def get_ignored_tokens(self, char_or_elem):
        beg_idx = self.get_beg_end_flag_idx("beg", char_or_elem)
        end_idx = self.get_beg_end_flag_idx("end", char_or_elem)
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end, char_or_elem):
        if char_or_elem == "char":
            if beg_or_end == "beg":
                idx = self.dict_character[self.beg_str]
            elif beg_or_end == "end":
                idx = self.dict_character[self.end_str]
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of char" \
                              % beg_or_end
        elif char_or_elem == "elem":
            if beg_or_end == "beg":
                idx = self.dict_elem[self.beg_str]
            elif beg_or_end == "end":
                idx = self.dict_elem[self.end_str]
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of elem" \
                              % beg_or_end
        else:
            assert False, "Unsupport type %s in char_or_elem" \
                          % char_or_elem
        return idx
