When you call /fetch_custom_adder, it triggers the following chain:

fetch_custom_adder_route()
   ↳ fetch_custom_adder()
      While not covered:
         ↳ select_best_adder(...)
            ↳ find_valid_start_positions()
            ↳ visualize_pyramid()
            ↳ (Makes request to OpenAI, parses JSON, etc.)
         If adder is returned:
            ↳ apply_adder_and_shrink_pyramid()
               (applies adder, logs, shrinks pyramid)
         Else:
            ↳ find_ones_heights()
            ↳ new_load_csv()
            ↳ find_closest_element_by_transformed_vector()
            ↳ generate_filled_pyramid()
            ↳ calculate_total_cost()
            (break loop)
      Return success/error
