import matplotlib.pyplot as plt
def plot_attention_head(in_tokens, translated_tokens, attention):
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label for label in in_tokens]
    ax.set_xticklabels(
        labels, rotation=90)

    labels = [label for label in translated_tokens]
    ax.set_yticklabels(labels)
    
def plot_attention_weights(sentence, translated_tokens, attention_heads, tokenizer, max_heads : int = 4):
    in_tokens = tokenizer.encode(sentence)
    in_tokens = tokenizer.convert_ids_to_tokens(in_tokens)
    
    translated_tokens = [label for label in translated_tokens.numpy()]
    translated_tokens = tokenizer.convert_ids_to_tokens(translated_tokens)
    
    fig = plt.figure(figsize=(12, 12))

    for h, head in enumerate(attention_heads[:max_heads]):
        ax = fig.add_subplot(2, 2, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    # plt.tight_layout()
    plt.show()    
