#define YOUNG_WORDS_BASE_OFFSET 0x1d0
#define DEST_REGIONATTR_OFFSET 0x178

#define PRE_DUMMY_TOP_OFFSET 0xa0

#define LHKID_OFFSET 8
#define VTABLE_LEN_OFFSET 160
#define ITABLE_LEN_OFFSET 296
#define TABLE_OFFSET 464
#define STATIC_COUNT_OFFSET

typedef struct RedirtyCardsLocalQueueSet{
    uintptr_t virtual_method_ptr;
    uintptr_t _allocator;
    uintptr_t _shared_qset;
    uintptr_t _buffers_head;
    uintptr_t _buffers_tail;
    size_t _buffers_entry_count;
    size_t _queue_index;
    size_t _queue_capacity_in_bytes;
    uintptr_t _queue_buf;
}RedirtyCardsLocalQueueSet;

typedef struct HeapRegionAttr{
    static const int8_t Num = 2;
    uint8_t _needs_remset_update;
    int8_t _type;
}HeapRegionAttr;

typedef struct ScanEvacuatedObjClosure __attribute__((aligned(8))){
    uintptr_t virtual_method_ptr;
    uintptr_t ref_discoverer;
    uintptr_t _g1h;
    uintptr_t _par_scan_state;
    uint32_t _scanning_in_young;
}

typedef struct CardTable{
    uintptr_t virtual_method_ptr;
    uintptr_t _whole_heap_start;
    size_t _whole_heap_word_size;
    size_t _guard_index;
    size_t _last_valid_index;
    size_t _page_size;
    size_t _byte_map_size;
    uintptr_t _byte_map;
    uintptr_t _byte_map_base;
    int _cur_covered_regions;
    uintptr_t _covered;
    uintptr_t _committed;
    uintptr_t _guard_region_heap_start;
    size_t _guard_region_word_size;
}CardTable;

typedef struct PLAB{
   char      head[32];
   size_t    _word_sz;
   uintptr_t _bottom;
   uintptr_t _top;
   uintptr_t _end;
   uintptr_t _hard_end;
   size_t    _allocated;
   size_t    _wasted;
   size_t    _undo_wasted;
   char      tail[32];
}

typedef struct PlabAllocator {
    uintptr_t _g1h;
    uintptr_t _allocator;
    PLAB** _alloc_buffers[G1HeapRegionAttr::Num];
}PlabAllocator;

typedef struct ParScanThreadState __attribute__((aligned(8))){
    uintptr_t _g1h;
    uintptr_t _task_queue;
    RedirtyCardsLocalQueueSet _rdc_local_qset;
    uintptr_t _ct;
    uintptr_t _closures;
    uintptr_t _plab_allocator;                  //0x70
    char [DEST_REGIONATTR_OFFSET -0x78];        //0x78
    HeapRegionAttr _dest[HeapRegionAttr::Num];  //0x178 179 17a 17b
    uint32_t _tenuring_threshold;               //0x17c
    ScanEvacuatedObjClosure _scanner;           //0x180
    uint32_t _worker_id;
    size_t _last_enqueued_card;                 // 0x1b0
    uint32_t const _stack_trim_upper_threshold; //0x1b8
    uint32_t const _stack_trim_lower_threshold; //0x1bc
    char [YOUNG_WORDS_BASE_OFFSET - 0x1c0];     //0x1c0
    uintptr_t _surviving_young_words_base;      //0x1d0
    uintptr_t _surviving_young_words;
    size_t  _surviving_words_length;
    bool _old_gen_is_full;
    int _partial_objarray_chunk_size;
 /*
    PartialArrayTaskStepper _partial_array_stepper;
    StringDedup::Requests _string_dedup_requests;
    size_t _num_optional_regions;
    uintptr_t _oops_into_optional_regions;
    uintptr_t _numa;
    uintptr_t _obj_alloc_stat;
 */
}

typedef struct BlockOffsetTablePart{
    uintptr_t _next_offset_threshold;
    uintptr_t _next_offset_index;
    uintptr_t _bot;
    uintptr_t _hr;
}BlockOffsetTablePart;

typedef struct HeapRegion{
    uintptr_t _bottom;
    uintptr_t _end;
    uintptr_t _top;
    uintptr_t _compaction_top;
    BlockOffsetTablePart _bot_part;     //0x20
    char [PRE_DUMMY_TOP_OFFSET - 0x40]; //0x40
    uintptr_t _pre_dummy_top;           //0xa8
    uintptr_t _rem_set;                 //0xb0
    uint32_t _hrm_index;                //0xb
    uint32_t _type;                     //0xbc
    uintptr_t _humongous_start_region;
    uint32_t _index_in_opt_cset;
    uintptr_t _next;
    uintptr_t _prev;
    uintptr_t _prev_top_at_mark_start;
    uintptr_t _next_top_at_mark_start;
    size_t _prev_marked_bytes;
    size_t _next_marked_bytes;
    uint32_t _young_index_in_cset;
    uintptr_t _sure_rate_group;
    int _age_index;
    double _gc_efficiency;
    uint32_t _node_index;
}HeapRegion;

typedef struct OopMap{
    int offset;
    int count;
}OopMap;

typedef struct Klass __attribute__((aligned(8))){
     char[LHKID_OFFSET];                //0
     uintptr_t lh_kid;                  //8
     char[VTABLE_LEN_OFFSET - 16];      //16
     int vtable_len;                    //160
     char[ITABLE_LEN_OFFSET - 164];     //164
     int itable_len;                    //296
     int nonStaticOopMapSize;           //300
     char[TABLE_OFFSET - 304];          //304
     uintptr_t paddings[];              //464
     OopMap maps[];                     //464 + (itable_len + vtable_len) * 8
}Klass;

typedef struct OopDesc __attribute__((aligned(8))){
    uintptr_t markWord;                 //0
    Klass * klass_ptr;                  //8
}OopDesc;

typedef struct ArrayOop __attribute__((aligned(8))) : public OopDesc{
    int array_length;                   //16
    uintptr_t contents[];               //24 elements_offset
}

typedef struct InstanceOop __attribute__((aligned(8))) : public OopDesc{
    uintptr_t contents[];
}

static inline size_t klass_size(int vtable_len, int itable_len, int oop_map_cnt){
    return TABLE_OFFSET +
           (size_t)(vtable_len + itable_len) * sizeof(uintptr_t) +
           (size_t)oop_map_cnt * sizeof(OopMap);
}

static inline uintptr_t* klass_tables(Klass* k) {
    return (uintptr_t*)((char*)k + TABLE_OFFSET);
}

static inline OopMap* klass_maps(Klass* k) {
    return (OopMap*)((char*)klass_tables(k) + (size_t)(k->vtable_len + k->itable_len) * sizeof(uintptr_t));
}

Klass* new_klass(int vtable_len, int itable_len, int oop_map_cnt) {
    size_t size = klass_size(vtable_len, itable_len, oop_map_cnt);
    Klass* k = aligned_alloc(8, size);

    memset(k, 0, TABLE_OFFSET);

    k->vtable_len = vtable_len;
    k->itable_len = itable_len;
    k->nonStaticOopMapSize = oop_map_cnt;

    return k;
}