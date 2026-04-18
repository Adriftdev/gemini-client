use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Attribute, Meta, ItemFn};

#[proc_macro_derive(GeminiSchema, attributes(gemini))]
pub fn derive_gemini_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    
    let description = get_description(&input.attrs);
    let description_token = match description {
        Some(d) => quote! { Some(#d.to_string()) },
        None => quote! { None },
    };

    let schema_gen = match input.data {
        Data::Struct(data) => {
            let fields = match data.fields {
                Fields::Named(fields) => fields.named,
                _ => panic!("GeminiSchema only supports structs with named fields"),
            };
            
            let prop_gen = fields.iter().map(|f| {
                let f_name = f.ident.as_ref().unwrap().to_string();
                let f_type = &f.ty;
                let f_desc = get_description(&f.attrs);
                let f_desc_token = match f_desc {
                    Some(d) => quote! { schema.description = Some(#d.to_string()); },
                    None => quote! {},
                };
                quote! {
                    {
                        let mut schema = <#f_type as ::gemini_client_rs::types::GeminiSchema>::schema();
                        #f_desc_token
                        properties.insert(#f_name.to_string(), schema);
                    }
                }
            });
            
            let required_gen = fields.iter().map(|f| {
                let f_name = f.ident.as_ref().unwrap().to_string();
                quote! { #f_name.to_string() }
            });

            quote! {
                let mut properties = std::collections::HashMap::new();
                #(#prop_gen)*
                
                ::gemini_client_rs::types::Schema {
                    schema_type: ::gemini_client_rs::types::SchemaType::Object,
                    description: #description_token,
                    properties: Some(properties),
                    required: Some(vec![#(#required_gen),*]),
                    ..Default::default()
                }
            }
        },
        _ => panic!("GeminiSchema only supports structs"),
    };

    let expanded = quote! {
        impl ::gemini_client_rs::types::GeminiSchema for #name {
            fn schema() -> ::gemini_client_rs::types::Schema {
                #schema_gen
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn gemini_tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_ident = &input_fn.sig.ident;
    let fn_name_str = fn_ident.to_string();
    
    let struct_name = format!("{}Tool", snake_to_camel(&fn_name_str));
    let struct_ident = syn::Ident::new(&struct_name, fn_ident.span());
    
    let description = get_description(&input_fn.attrs).unwrap_or_default();
    
    let mut param_props = quote! {};
    let mut required_params = quote! {};
    
    for arg in &input_fn.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let arg_name = pat_ident.ident.to_string();
                let arg_type = &pat_type.ty;
                param_props = quote! {
                    #param_props
                    properties.insert(#arg_name.to_string(), <#arg_type as ::gemini_client_rs::types::GeminiSchema>::schema());
                };
                required_params = quote! {
                    #required_params
                    required.push(#arg_name.to_string());
                };
            }
        }
    }

    let expanded = quote! {
        #input_fn

        pub struct #struct_ident;

        impl ::gemini_client_rs::types::GeminiTool for #struct_ident {
            fn declaration() -> ::gemini_client_rs::types::FunctionDeclaration {
                let mut properties = std::collections::HashMap::new();
                let mut required = Vec::new();
                #param_props
                
                ::gemini_client_rs::types::FunctionDeclaration {
                    name: #fn_name_str.to_string(),
                    description: #description.to_string(),
                    parameters: Some(::gemini_client_rs::types::Schema {
                        schema_type: ::gemini_client_rs::types::SchemaType::Object,
                        properties: Some(properties),
                        required: Some(required),
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            }
        }
    };

    TokenStream::from(expanded)
}

fn get_description(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &nv.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        return Some(lit_str.value().trim().to_string());
                    }
                }
            }
        }
    }
    None
}

fn snake_to_camel(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}
