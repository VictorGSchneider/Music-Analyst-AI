#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "uthash.h"

#define TAG_LYRIC 100
#define TAG_WORD 200
#define TAG_ARTIST 300

#define MAX_TOP_ITEMS 20

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct word_entry {
    char *word;
    long long count;
    UT_hash_handle hh;
} word_entry;

typedef struct artist_entry {
    char *artist;
    long long count;
    UT_hash_handle hh;
} artist_entry;

static void free_word_map(word_entry **map) {
    word_entry *entry, *tmp;
    HASH_ITER(hh, *map, entry, tmp) {
        HASH_DEL(*map, entry);
        free(entry->word);
        free(entry);
    }
}

static void free_artist_map(artist_entry **map) {
    artist_entry *entry, *tmp;
    HASH_ITER(hh, *map, entry, tmp) {
        HASH_DEL(*map, entry);
        free(entry->artist);
        free(entry);
    }
}

static void add_word(word_entry **map, const char *word, long long value) {
    if (word[0] == '\0') {
        return;
    }
    word_entry *entry = NULL;
    HASH_FIND_STR(*map, word, entry);
    if (entry == NULL) {
        entry = (word_entry *)malloc(sizeof(word_entry));
        if (!entry) {
            fprintf(stderr, "Failed to allocate memory for word entry\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        entry->word = strdup(word);
        entry->count = value;
        HASH_ADD_KEYPTR(hh, *map, entry->word, strlen(entry->word), entry);
    } else {
        entry->count += value;
    }
}

static void add_artist(artist_entry **map, const char *artist, long long value) {
    if (artist[0] == '\0') {
        return;
    }
    artist_entry *entry = NULL;
    HASH_FIND_STR(*map, artist, entry);
    if (entry == NULL) {
        entry = (artist_entry *)malloc(sizeof(artist_entry));
        if (!entry) {
            fprintf(stderr, "Failed to allocate memory for artist entry\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        entry->artist = strdup(artist);
        entry->count = value;
        HASH_ADD_KEYPTR(hh, *map, entry->artist, strlen(entry->artist), entry);
    } else {
        entry->count += value;
    }
}

static void sanitize_and_write_lyric(FILE *fp, const char *text) {
    for (const char *ptr = text; *ptr; ++ptr) {
        char c = *ptr;
        if (c == '\r' || c == '\n') {
            fputc(' ', fp);
        } else {
            fputc(c, fp);
        }
    }
    fputc('\n', fp);
}

static void update_word_counts(word_entry **map, const char *text) {
    char buffer[4096];
    size_t buf_len = 0;
    size_t text_len = strlen(text);

    for (size_t i = 0; i <= text_len; ++i) {
        char c = (i < text_len) ? text[i] : ' ';
        if (isalnum((unsigned char)c)) {
            if (buf_len + 1 >= sizeof(buffer)) {
                buffer[buf_len] = '\0';
                for (size_t j = 0; j < buf_len; ++j) {
                    buffer[j] = (char)tolower((unsigned char)buffer[j]);
                }
                add_word(map, buffer, 1);
                buf_len = 0;
            }
            buffer[buf_len++] = (char)tolower((unsigned char)c);
        } else {
            if (buf_len > 0) {
                buffer[buf_len] = '\0';
                add_word(map, buffer, 1);
                buf_len = 0;
            }
        }
    }
}

static int parse_csv_record(const char *record, char **artist, char **song, char **lyrics) {
    const size_t len = strlen(record);
    char *field_buffer = (char *)malloc(len + 1);
    if (!field_buffer) {
        return -1;
    }
    char *fields[4] = {NULL};
    size_t field_index = 0;
    size_t buf_len = 0;
    int in_quotes = 0;

    for (size_t i = 0; i < len; ++i) {
        char c = record[i];
        if (c == '\r') {
            continue;
        }
        if (c == '"') {
            if (in_quotes && i + 1 < len && record[i + 1] == '"') {
                field_buffer[buf_len++] = '"';
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            field_buffer[buf_len] = '\0';
            fields[field_index++] = strdup(field_buffer);
            buf_len = 0;
            if (field_index >= 4) {
                break;
            }
        } else {
            field_buffer[buf_len++] = c;
        }
    }

    if (field_index < 4 && buf_len > 0) {
        field_buffer[buf_len] = '\0';
        fields[field_index++] = strdup(field_buffer);
    }

    free(field_buffer);

    if (field_index != 4) {
        for (size_t i = 0; i < field_index; ++i) {
            free(fields[i]);
        }
        return -1;
    }

    *artist = fields[0];
    *song = fields[1];
    *lyrics = fields[3];
    free(fields[2]);
    return 0;
}

static int read_csv_record(FILE *fp, char **artist, char **song, char **lyrics) {
    char *line = NULL;
    size_t capacity = 0;
    ssize_t read = 0;
    size_t total_len = 0;
    size_t quote_count = 0;
    char *record_buffer = NULL;

    while ((read = getline(&line, &capacity, fp)) != -1) {
        record_buffer = (char *)realloc(record_buffer, total_len + read + 1);
        if (!record_buffer) {
            free(line);
            return -1;
        }
        memcpy(record_buffer + total_len, line, (size_t)read);
        total_len += (size_t)read;
        record_buffer[total_len] = '\0';

        for (ssize_t i = 0; i < read; ++i) {
            if (line[i] == '"') {
                ++quote_count;
            }
        }
        if (quote_count % 2 == 0) {
            break;
        }
    }

    free(line);

    if (total_len == 0) {
        free(record_buffer);
        return -1;
    }

    int result = parse_csv_record(record_buffer, artist, song, lyrics);
    free(record_buffer);
    return result;
}

static void process_record(word_entry **word_map, artist_entry **artist_map, FILE *tmp_fp,
                           const char *artist, const char *lyrics) {
    if (artist && artist[0]) {
        add_artist(artist_map, artist, 1);
    }
    if (lyrics && lyrics[0]) {
        update_word_counts(word_map, lyrics);
        if (tmp_fp) {
            sanitize_and_write_lyric(tmp_fp, lyrics);
        }
    }
}

static void send_termination_signal(int dest, int tag) {
    int end = -1;
    MPI_Send(&end, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
}

static void send_word_map_to_root(word_entry *map) {
    word_entry *entry, *tmp;
    HASH_ITER(hh, map, entry, tmp) {
        int len = (int)strlen(entry->word);
        MPI_Send(&len, 1, MPI_INT, 0, TAG_WORD, MPI_COMM_WORLD);
        MPI_Send(entry->word, len, MPI_CHAR, 0, TAG_WORD, MPI_COMM_WORLD);
        MPI_Send(&entry->count, 1, MPI_LONG_LONG, 0, TAG_WORD, MPI_COMM_WORLD);
    }
    send_termination_signal(0, TAG_WORD);
}

static void send_artist_map_to_root(artist_entry *map) {
    artist_entry *entry, *tmp;
    HASH_ITER(hh, map, entry, tmp) {
        int len = (int)strlen(entry->artist);
        MPI_Send(&len, 1, MPI_INT, 0, TAG_ARTIST, MPI_COMM_WORLD);
        MPI_Send(entry->artist, len, MPI_CHAR, 0, TAG_ARTIST, MPI_COMM_WORLD);
        MPI_Send(&entry->count, 1, MPI_LONG_LONG, 0, TAG_ARTIST, MPI_COMM_WORLD);
    }
    send_termination_signal(0, TAG_ARTIST);
}

static void merge_word_entry(word_entry **map, const char *word, long long count) {
    add_word(map, word, count);
}

static void merge_artist_entry(artist_entry **map, const char *artist, long long count) {
    add_artist(map, artist, count);
}

static int compare_word_entries(const void *a, const void *b) {
    const word_entry *entry_a = *(const word_entry **)a;
    const word_entry *entry_b = *(const word_entry **)b;
    if (entry_a->count == entry_b->count) {
        return strcmp(entry_a->word, entry_b->word);
    }
    return (entry_b->count > entry_a->count) - (entry_b->count < entry_a->count);
}

static int compare_artist_entries(const void *a, const void *b) {
    const artist_entry *entry_a = *(const artist_entry **)a;
    const artist_entry *entry_b = *(const artist_entry **)b;
    if (entry_a->count == entry_b->count) {
        return strcmp(entry_a->artist, entry_b->artist);
    }
    return (entry_b->count > entry_a->count) - (entry_b->count < entry_a->count);
}

static void print_top_words(word_entry *map) {
    unsigned long count = HASH_COUNT(map);
    word_entry **entries = (word_entry **)malloc(count * sizeof(word_entry *));
    if (!entries) {
        fprintf(stderr, "Failed to allocate memory for top words\n");
        return;
    }
    word_entry *entry;
    unsigned long idx = 0;
    for (entry = map; entry != NULL; entry = entry->hh.next) {
        entries[idx++] = entry;
    }
    qsort(entries, count, sizeof(word_entry *), compare_word_entries);
    unsigned long limit = count < MAX_TOP_ITEMS ? count : MAX_TOP_ITEMS;
    printf("\nTop %lu palavras:\n", limit);
    for (unsigned long i = 0; i < limit; ++i) {
        printf("%2lu. %-25s %lld\n", i + 1, entries[i]->word, entries[i]->count);
    }
    free(entries);
}

static void print_top_artists(artist_entry *map) {
    unsigned long count = HASH_COUNT(map);
    artist_entry **entries = (artist_entry **)malloc(count * sizeof(artist_entry *));
    if (!entries) {
        fprintf(stderr, "Failed to allocate memory for top artists\n");
        return;
    }
    artist_entry *entry;
    unsigned long idx = 0;
    for (entry = map; entry != NULL; entry = entry->hh.next) {
        entries[idx++] = entry;
    }
    qsort(entries, count, sizeof(artist_entry *), compare_artist_entries);
    unsigned long limit = count < MAX_TOP_ITEMS ? count : MAX_TOP_ITEMS;
    printf("\nTop %lu artistas por quantidade de músicas:\n", limit);
    for (unsigned long i = 0; i < limit; ++i) {
        printf("%2lu. %-30s %lld\n", i + 1, entries[i]->artist, entries[i]->count);
    }
    free(entries);
}

static int classify_chunk(const char *tmp_path, long long *positive, long long *neutral, long long *negative) {
    char command[PATH_MAX];
    snprintf(command, sizeof(command), "python3 sentiment_classifier.py --input \"%s\"", tmp_path);
    FILE *pipe = popen(command, "r");
    if (!pipe) {
        fprintf(stderr, "Failed to execute sentiment classifier script.\n");
        return -1;
    }
    char buffer[256];
    if (!fgets(buffer, sizeof(buffer), pipe)) {
        pclose(pipe);
        fprintf(stderr, "No output from sentiment classifier script.\n");
        return -1;
    }
    int status = pclose(pipe);
    if (status == -1) {
        fprintf(stderr, "Failed to close sentiment classifier pipe.\n");
        return -1;
    }

    long long pos = 0, neu = 0, neg = 0;
    if (sscanf(buffer, "%lld %lld %lld", &pos, &neu, &neg) != 3) {
        fprintf(stderr, "Unexpected classifier output: %s\n", buffer);
        return -1;
    }

    *positive = pos;
    *neutral = neu;
    *negative = neg;
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 1) {
        fprintf(stderr, "At least one MPI process is required.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const char *csv_path = "spotify_millsongdata.csv";
    if (argc > 1) {
        csv_path = argv[1];
    }

    word_entry *word_map = NULL;
    artist_entry *artist_map = NULL;

    char tmp_path[PATH_MAX];
    snprintf(tmp_path, sizeof(tmp_path), "classification_rank_%d.txt", world_rank);
    FILE *tmp_fp = fopen(tmp_path, "w");
    if (!tmp_fp) {
        fprintf(stderr, "Processo %d não conseguiu criar arquivo temporário %s: %s\n",
                world_rank, tmp_path, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double start_time = MPI_Wtime();

    if (world_rank == 0) {
        FILE *fp = fopen(csv_path, "r");
        if (!fp) {
            fprintf(stderr, "Não foi possível abrir %s: %s\n", csv_path, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        char *artist = NULL;
        char *song = NULL;
        char *lyrics = NULL;

        char header[1024];
        if (!fgets(header, sizeof(header), fp)) {
            fprintf(stderr, "Arquivo CSV vazio ou inválido.\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        long long record_index = 0;
        while (read_csv_record(fp, &artist, &song, &lyrics) == 0) {
            int dest = (int)(record_index % world_size);
            if (dest == 0) {
                process_record(&word_map, &artist_map, tmp_fp, artist, lyrics);
            } else {
                int artist_len = (int)strlen(artist);
                int lyric_len = (int)strlen(lyrics);
                MPI_Send(&lyric_len, 1, MPI_INT, dest, TAG_LYRIC, MPI_COMM_WORLD);
                MPI_Send(&artist_len, 1, MPI_INT, dest, TAG_LYRIC, MPI_COMM_WORLD);
                if (lyric_len > 0) {
                    MPI_Send(lyrics, lyric_len, MPI_CHAR, dest, TAG_LYRIC, MPI_COMM_WORLD);
                }
                if (artist_len > 0) {
                    MPI_Send(artist, artist_len, MPI_CHAR, dest, TAG_LYRIC, MPI_COMM_WORLD);
                }
            }
            free(artist);
            free(song);
            free(lyrics);
            artist = song = lyrics = NULL;
            ++record_index;
        }
        fclose(fp);

        for (int dest = 1; dest < world_size; ++dest) {
            send_termination_signal(dest, TAG_LYRIC);
        }
    } else {
        MPI_Status status;
        while (1) {
            int lyric_len = 0;
            MPI_Recv(&lyric_len, 1, MPI_INT, 0, TAG_LYRIC, MPI_COMM_WORLD, &status);
            if (lyric_len == -1) {
                break;
            }
            int artist_len = 0;
            MPI_Recv(&artist_len, 1, MPI_INT, 0, TAG_LYRIC, MPI_COMM_WORLD, &status);

            char *lyric_buf = NULL;
            char *artist_buf = NULL;
            if (lyric_len > 0) {
                lyric_buf = (char *)malloc((size_t)lyric_len + 1);
                MPI_Recv(lyric_buf, lyric_len, MPI_CHAR, 0, TAG_LYRIC, MPI_COMM_WORLD, &status);
                lyric_buf[lyric_len] = '\0';
            } else {
                lyric_buf = strdup("");
            }
            if (artist_len > 0) {
                artist_buf = (char *)malloc((size_t)artist_len + 1);
                MPI_Recv(artist_buf, artist_len, MPI_CHAR, 0, TAG_LYRIC, MPI_COMM_WORLD, &status);
                artist_buf[artist_len] = '\0';
            } else {
                artist_buf = strdup("");
            }

            process_record(&word_map, &artist_map, tmp_fp, artist_buf, lyric_buf);

            free(lyric_buf);
            free(artist_buf);
        }
    }

    fflush(tmp_fp);
    fclose(tmp_fp);

    double processing_end = MPI_Wtime();

    long long local_positive = 0;
    long long local_neutral = 0;
    long long local_negative = 0;

    double classification_start = MPI_Wtime();
    if (classify_chunk(tmp_path, &local_positive, &local_neutral, &local_negative) != 0) {
        fprintf(stderr, "Processo %d não conseguiu classificar o lote de letras.\n", world_rank);
    }
    double classification_end = MPI_Wtime();

    remove(tmp_path);

    long long global_positive = 0;
    long long global_neutral = 0;
    long long global_negative = 0;

    MPI_Reduce(&local_positive, &global_positive, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_neutral, &global_neutral, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_negative, &global_negative, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double processing_time = processing_end - start_time;
    double classification_time = classification_end - classification_start;

    double max_processing_time = 0.0;
    double avg_processing_time = 0.0;
    double max_classification_time = 0.0;
    double avg_classification_time = 0.0;

    MPI_Reduce(&processing_time, &max_processing_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&processing_time, &avg_processing_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&classification_time, &max_classification_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&classification_time, &avg_classification_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        avg_processing_time /= world_size;
        avg_classification_time /= world_size;

        for (int source = 1; source < world_size; ++source) {
            MPI_Status status;
            while (1) {
                int len = 0;
                MPI_Recv(&len, 1, MPI_INT, source, TAG_WORD, MPI_COMM_WORLD, &status);
                if (len == -1) {
                    break;
                }
                char *word = (char *)malloc((size_t)len + 1);
                MPI_Recv(word, len, MPI_CHAR, source, TAG_WORD, MPI_COMM_WORLD, &status);
                word[len] = '\0';
                long long count = 0;
                MPI_Recv(&count, 1, MPI_LONG_LONG, source, TAG_WORD, MPI_COMM_WORLD, &status);
                merge_word_entry(&word_map, word, count);
                free(word);
            }
        }

        for (int source = 1; source < world_size; ++source) {
            MPI_Status status;
            while (1) {
                int len = 0;
                MPI_Recv(&len, 1, MPI_INT, source, TAG_ARTIST, MPI_COMM_WORLD, &status);
                if (len == -1) {
                    break;
                }
                char *artist = (char *)malloc((size_t)len + 1);
                MPI_Recv(artist, len, MPI_CHAR, source, TAG_ARTIST, MPI_COMM_WORLD, &status);
                artist[len] = '\0';
                long long count = 0;
                MPI_Recv(&count, 1, MPI_LONG_LONG, source, TAG_ARTIST, MPI_COMM_WORLD, &status);
                merge_artist_entry(&artist_map, artist, count);
                free(artist);
            }
        }

        print_top_words(word_map);
        print_top_artists(artist_map);

        printf("\nClassificação de sentimentos (total):\n");
        printf("  Positivas: %lld\n", global_positive);
        printf("  Neutras:   %lld\n", global_neutral);
        printf("  Negativas: %lld\n", global_negative);

        printf("\nMétricas de desempenho:\n");
        printf("  Tempo médio de processamento (palavras/artistas): %.4f s\n", avg_processing_time);
        printf("  Tempo máximo de processamento: %.4f s\n", max_processing_time);
        printf("  Tempo médio de classificação: %.4f s\n", avg_classification_time);
        printf("  Tempo máximo de classificação: %.4f s\n", max_classification_time);
    } else {
        send_word_map_to_root(word_map);
        send_artist_map_to_root(artist_map);
    }

    free_word_map(&word_map);
    free_artist_map(&artist_map);

    MPI_Finalize();
    return 0;
}
