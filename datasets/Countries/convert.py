def load_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            name, id = line.strip().split('\t')
            mapping[name] = int(id)
    return mapping

def convert_triples(input_file, output_file, entity_map, relation_map):
    triplets = []
    with open(input_file, 'r') as infile:
        for line in infile:
            head, relation, tail = line.strip().split('\t')
            head_id = entity_map.get(head, -1)
            relation_id = relation_map.get(relation, -1)
            tail_id = entity_map.get(tail, -1)
            if head_id == -1 or relation_id == -1 or tail_id == -1:
                raise ValueError(f"Unknown entity or relation in line: {line.strip()}")
            triplets.append((head_id, tail_id, relation_id))

    with open(output_file, 'w') as outfile:
        outfile.write(f"{len(triplets)}\n")
        for head_id, tail_id, relation_id in triplets:
            outfile.write(f"{head_id}\t{tail_id}\t{relation_id}\n")

def main():
    entity2id = load_mapping('entity2id.txt')
    relation2id = load_mapping('relation2id.txt')
    
    convert_triples('train.txt', 'train2id.txt', entity2id, relation2id)
    convert_triples('test.txt', 'test2id.txt', entity2id, relation2id)
    convert_triples('valid.txt', 'valid2id.txt', entity2id, relation2id)

if __name__ == '__main__':
    main()
