

# API Reference


::: pulseseek
    options:
      members: true                 # include package-level members
      members_order: source
      group_by_category: true
      show_submodules: true         # list submodules in a "Submodules" section
      show_root_toc_entry: false
      show_source: true
      docstring_style: google       # or numpy
      merge_init_into_class: true
      inherited_members: true
      filters:
        - "!^_"                     # hide private names