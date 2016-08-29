flask-docjson
=============

Validate flask request and response json schemas via docstring.

```Python
@app.route('/user/<id>', methods=['GET'])
def get_user(id):
    """Schema::

        GET /user/<i32:id>
        200
        {"id": i32, "name": string(32)}
    """
```

Purpose
-------

* Bye: repetitive json type validation work.
* Hello: concise flask api documentation.


Example
-------

**Strongly recommend** that you checkout [example.py](example.py) at first.

Install
-------

```
pip install flask-docjson
```

Usage
-----

*  If you want to validate all routes via flask-docjson:

   ```python
   # Must be called after all routes are defined.
   flask_docjson.register_all(app)
   ```

*  If you don't want to validate all routes, just use the decorator instead:

   ```python
   @app.route('/user/<id>', methods=['GET'])
   @flask_docjson.validate
   def get_user(id):
       """Schema::
           ...
       """
   ```

Base Types
----------

- Bool: `bool`
- Number: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `float`
- String: `string`, `string(maxlength)` (e.g. `string(32)`)

Containers
----------

- Array: e.g. `[u8, i32, string]`, `[u8, ...]`
- Object: e.g. `{"name": string, items: [u8, ...]}`

Ellipsis Array
--------------

For an example schema::

```
{
    "items": [u8, ...]
}
```

1. Empty `list` (aka `[]`) passes this schema.
2. For a given `list`, all its elements should be `u8`.

Ellipsis Object
---------------

```
{
    "id": i32,
    ...
}
```

An object with ellipsis is exactly the same with this object without ellipsis, 
this feature is useful to indicate this object may contains more fields, but
the author just don't want to validate or document them.

Nullable Value
--------------

By default, values shouldn't be nullable, an example to change this behavior:

```
{
    "optional-string": string(32)*,
    "optional-number": i32*
}
```

But `*` can't be used with the whole JSON schema.

No Content Request/Response
----------------------------

Just leave the schema blank:

* Request schema without content example:

   ```
   GET /data
   200
   {"key": i32}
   ```

* Response schema without content example:

   ```
   DELETE /data/<i32:id>
   201
   ```

Minor Differences from JSON
---------------------------

In standard JSON, schema with a trailing comma (e.g. `{"key": "value",}`) is illegal,
but our JSON schema allows this. Because that makes modifications by line convenience.

Language Description
--------------------

* Document: `Schema ::= Request Response*`.
* Request: `Request ::= Method* Route JsonSchema`
* Response: `Response ::= StatusCode* JsonSchema`
* Method: `Method ::= POST|PUT|GET|DELETE|PATCH|HEAD|OPTIONS`. Method separator is `/`.
* Route: `Route ::= StaticRoute* RouteVar* StaticRoute*`. e.g. `/user/<i32:id>`
* StatusCode: `StatusCode ::= Integer|Matcher`. e.g. `200`, `4XX`
* JsonSchema: `JsonSchema ::= Array | Object`.
* Array: `Array ::= '[' Value* | Value* '...' ']'`
* Object: `Object ::= '{' KeyValue* '}'`
* KeyValue: `KeyValue ::= Key ':' Value`
* Value: `Array ::= BaseType | Array | Object`

License
-------
BSD.
