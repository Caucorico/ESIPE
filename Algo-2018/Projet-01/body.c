#include "body.h"

body* create_body( int x, int y, float speed_x, float speed_y, float acceleration_x, float acceleration_y, float mass )
{
	body* new_body = malloc(sizeof(body));
	if ( body == NULL )return NULL;

	new_body->x = x;
	new_body->y = y;
	new_body->mass = mass;

	new_body->av = create_acceleration_vector();
	if ( new_body->av == NULL )
	{
		free(new_body);
		return NULL;
	}
	else
	{
		new_body->av->acceleration_x = acceleration_x;
		new_body->av->acceleration_y = acceleration_y;
	}

	new_body->sv = create_speed_vector();
	if ( new_body->sv == NULL )
	{
		free(new_body);
		return NULL;
	}
	else
	{
		new_body->av->speed_x = speed_x;
		new_body->av->speed_y = speed_y;
	}

	return new_body;
}

void free_body( body* b )
{
	if ( b != NULL )
	{
		if ( b->av != NULL ) free(b->av);
		if ( b->sv != NULL ) free(b->sv);

		free(b);
	}
}