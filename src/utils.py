import pandas as pd
import sqlite3
import torch
import torch.nn.functional as F
import os
import re

def get_total_power(outputs, t1, t2, fname):
    if fname is not None:
        with open(fname, 'wb') as f:
            pickle.dump((outputs, t1, t2), f)
    x = [out.strip().split() for out in outputs]
    x = [[float(xx[0]), float(xx[1])] for xx in x if len(xx) >= 2] # it seems possible that the last output of nvidia-smi is missing
    total_power = 0
    first_one = True
    for timestamp, power in x:
        if timestamp > t1 and timestamp < t2:
            if first_one:
                first_one = False
            else:
                total_power += power
    return total_power


def get_score(output, target_model, input_len):
    with torch.no_grad():
        if target_model.config.is_encoder_decoder == False:
            logits = target_model(output).logits
            logits = logits[:,:-1,:]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = torch.gather(logits,
                          dim = -1,
                          index = output[:,1:,None])
            if logits.isnan().any():
                print(logits.size())
                print(logits)
                print(old_logits)
                xxx = input()

            return torch.mean(logits[:,input_len-1:,:])
        else:
            logits = target_model(output[:, :input_len], decoder_input_ids=output[:,input_len:]).logits
            logits = logits[:, :-1, :]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = torch.gather(logits,
                                  dim = -1,
                                  index = output[:, input_len+1:, None])
            return torch.mean(logits)


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """

    if top_k is not None and top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[..., [-1]]] = float('-inf')
    if top_p is not None and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        dim = filter.dim() - 1 
        indices_to_remove = filter.scatter(dim, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    ori_logits = logits
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = torch.log_softmax(logits, dim=-1).exp()
    if probs.isnan().any() or probs.isinf().any() or (probs<0).any():
        print(torch.logical_not(logits.isinf()).any())
        print(logits[probs.isnan()])
        print(logits[probs.isinf()])
        raise RuntimeError('norm logits error')


    return probs

spider_examples = """[INST] <<SYS>> You are a SQL expert. You need to write the correct SQL based on the user question and database schemas. Below are some examples <</SYS>>
Example
Schema:
Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Question: "How many heads of the departments are older than 56 ?"
SQL: SELECT count(*) FROM head WHERE age  >  56; 

"""

def find_fields_MYSQL_like(db_name, spider_schema):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output

def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def match(result1, result2):
    str_result1 = set()
    str_result2 = set()
    for row in result1:
        str_row = [str(c) for c in row]
        str_row.sort()
        str_result1.add(tuple(str_row))
    for row in result2:
        str_row = [str(c) for c in row]
        str_row.sort()
        str_result2.add(tuple(str_row))

    return str_result1 == str_result2

def execution_accuracy(db, pred, sql, data_path):
    conn = sqlite3.connect(os.path.join(data_path, f"database/{db}/{db}.sqlite"))
    conn.text_factory = bytes
    cur = conn.cursor()
    try:
        gt_result = cur.execute(sql).fetchall()
    except Exception as e:
        return -1

    try:
        result = cur.execute(pred).fetchall()
    except Exception as e:
        return 0
    spider_acc = float(match(result, gt_result))
    return spider_acc


def execution_accuracy_references(predictions, references, data_path):
    em = 0
    exception = 0
    for pred, refer in zip(predictions, references):
        db = refer.split("[SQL]")[0]
        sql = refer.split("[SQL]")[1]
        acc = execution_accuracy(db, pred, sql, data_path)
        if acc >= 0:
            em += acc
        else:
            exception += 1
    return {"execution accuracy": 100.0 * em / (len(predictions)-exception), "exception": exception}

def extract_first_function(code_string):
    """
    Extract the first function definition from a Python code string.
    Handles imports, docstrings, and type hints.
    
    Args:
        code_string (str): String containing Python code
        
    Returns:
        str: The first function definition found, or empty string if none found
    """
    
    # Split code into lines while preserving empty lines
    lines = code_string.split('\n')
    
    # Initialize variables
    function_lines = []
    in_function = False
    base_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and imports before function
        if not in_function and (not stripped or stripped.startswith('from ') or stripped.startswith('import ')):
            if stripped.startswith(('from ', 'import ')):
                function_lines.append(line)
            i += 1
            continue
            
        # Check for function definition
        if not in_function and re.match(r'^\s*def\s+\w+.*:', line):
            in_function = True
            base_indent = len(line) - len(line.lstrip())
            function_lines.append(line)
            i += 1
            continue
            
        # If we're in a function
        if in_function:
            # Skip empty lines
            if not stripped:
                function_lines.append(line)
                i += 1
                continue
                
            current_indent = len(line) - len(line.lstrip())
            # If we hit a line with same or less indent (except empty lines), we're done
            if current_indent <= base_indent and stripped:
                break
                
            function_lines.append(line)
            i += 1
            continue
            
        i += 1
    
    return '\n'.join(function_lines)

